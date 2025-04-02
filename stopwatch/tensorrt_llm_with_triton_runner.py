from typing import Any, Mapping, Optional
import contextlib
import hashlib
import json
import os
import subprocess
import time
import warnings

import modal

from .resources import app, hf_cache_volume, hf_secret


HF_CACHE_PATH = "/cache"
SCALEDOWN_WINDOW = 30  # 30 seconds
STARTUP_TIMEOUT = 30 * 60  # 30 minutes
TIMEOUT = 60 * 60  # 1 hour


def tensorrt_llm_image_factory(
    tensorrt_llm_version: str = "0.17.0.post1", cuda_version: str = "12.8.0"
):
    return (
        modal.Image.from_registry(
            "nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3", add_python="3.12"
        )
        .run_commands(
            "mv /usr/local/bin/python /usr/local/bin/python.modal",
            "mv /usr/local/bin/python3 /usr/local/bin/python3.modal",
            "mkdir /triton_model_repo",
            "git clone https://github.com/triton-inference-server/tensorrtllm_backend /opt/tensorrtllm_backend",
            "cp -r /opt/tensorrtllm_backend/all_models/inflight_batcher_llm/* /triton_model_repo/",
        )
        .pip_install(
            f"tensorrt-llm=={tensorrt_llm_version}",
            "pynvml<12",  # avoid breaking change to pynvml version API
            pre=True,
            extra_index_url="https://pypi.nvidia.com",
        )
        .pip_install(
            "hf-transfer",
            "huggingface-hub",
            "grpclib",
            "numpy",
            "requests",
            "SQLAlchemy",
        )
        .env(
            {
                "HF_HUB_CACHE": HF_CACHE_PATH,
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            }
        )
    )


def tensorrt_llm_cls(
    image=tensorrt_llm_image_factory(),
    secrets=[hf_secret],
    gpu="H100!",
    volumes={HF_CACHE_PATH: hf_cache_volume},
    cpu=4,
    memory=65536,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT,
    region="us-chicago-1",
):
    def decorator(cls):
        return app.cls(
            image=image,
            secrets=secrets,
            gpu=gpu,
            volumes=volumes,
            cpu=cpu,
            memory=memory,
            max_containers=1,
            allow_concurrent_inputs=1000,  # Set to a high number to prevent auto-scaling
            scaledown_window=scaledown_window,
            timeout=timeout,
            region=region,
        )(cls)

    return decorator


class TensorRTLLMWithTritonBase:
    """
    A Modal class that runs an OpenAI-compatible TensorRT-LLM server.
    """

    def construct_engine_kwargs(self) -> dict:
        import tensorrt_llm
        from tensorrt_llm import BuildConfig
        from tensorrt_llm.llmapi import (
            QuantConfig,
            KvCacheConfig,
            CalibConfig,
            LookaheadDecodingConfig,
        )
        from tensorrt_llm.plugin.plugin import PluginConfig

        config_name_to_constructor = {
            "quant_config": lambda x: QuantConfig(**x),
            "kv_cache_config": lambda x: KvCacheConfig(**x),
            "calib_config": lambda x: CalibConfig(**x),
            "build_config": lambda x: BuildConfig(**x),
            "plugin_config": lambda x: PluginConfig.from_dict(x),
            # TODO: SchedulerConfig
            # TODO: Add support for other decoding configs
            "speculative_config": lambda x: LookaheadDecodingConfig(**x),
        }

        llm_kwargs = json.loads(self.server_config).get("llm_kwargs", {})

        self.config_fingerprint = self.get_config_fingerprint(
            tensorrt_llm.__version__, llm_kwargs
        )

        def construct_configs(x):
            for key, value in x.items():
                if isinstance(value, dict):
                    x[key] = construct_configs(value)
                if key in config_name_to_constructor:
                    x[key] = config_name_to_constructor[key](value)
            return x

        llm_kwargs = construct_configs(llm_kwargs)
        self.lookahead_config = llm_kwargs.get("speculative_config")

        return llm_kwargs

    def build_engine(self, engine_path, engine_kwargs) -> None:
        from tensorrt_llm import LLM

        print(f"building new engine at {engine_path}")
        llm = LLM(model=self.model_path, **engine_kwargs)
        llm.save(engine_path)
        del llm

    @modal.enter()
    def enter(self):
        from huggingface_hub import snapshot_download

        print("Downloading base model if necessary")
        self.model_path = snapshot_download(self.model)

        engine_kwargs = self.construct_engine_kwargs()

        # FIXME just make this trttlm version-engine
        engine_path = os.path.join(
            self.model_path, "tensorrt_llm_with_triton_engine", self.config_fingerprint
        )
        if not os.path.exists(engine_path):
            self.build_engine(engine_path, engine_kwargs)

        ENGINE_DIR = engine_path
        TOKENIZER_DIR = self.model_path
        MODEL_FOLDER = "/triton_model_repo"
        TRITON_MAX_BATCH_SIZE = 4
        INSTANCE_COUNT = 1
        MAX_QUEUE_DELAY_MS = 0
        MAX_QUEUE_SIZE = 0
        FILL_TEMPLATE_SCRIPT = "/opt/tensorrtllm_backend/tools/fill_template.py"
        DECOUPLED_MODE = "true"
        LOGITS_DATATYPE = "TYPE_FP32"
        ENCODER_INPUT_FEATURES_DATA_TYPE = "TYPE_BF16"

        os.system(
            f"/usr/bin/python {FILL_TEMPLATE_SCRIPT} -i {MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:{TRITON_MAX_BATCH_SIZE},logits_datatype:{LOGITS_DATATYPE}"
        )
        os.system(
            f"/usr/bin/python {FILL_TEMPLATE_SCRIPT} -i {MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:{TOKENIZER_DIR},triton_max_batch_size:{TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:{INSTANCE_COUNT}"
        )
        os.system(
            f"/usr/bin/python {FILL_TEMPLATE_SCRIPT} -i {MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:{TRITON_MAX_BATCH_SIZE},decoupled_mode:{DECOUPLED_MODE},engine_dir:{ENGINE_DIR},max_queue_delay_microseconds:{MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,max_queue_size:{MAX_QUEUE_SIZE},encoder_input_features_data_type:{ENCODER_INPUT_FEATURES_DATA_TYPE},logits_datatype:{LOGITS_DATATYPE}"
        )
        os.system(
            f"/usr/bin/python {FILL_TEMPLATE_SCRIPT} -i {MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:{TOKENIZER_DIR},triton_max_batch_size:{TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:{INSTANCE_COUNT}"
        )
        os.system(
            f"/usr/bin/python {FILL_TEMPLATE_SCRIPT} -i {MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:{TRITON_MAX_BATCH_SIZE},decoupled_mode:{DECOUPLED_MODE},bls_instance_count:{INSTANCE_COUNT},logits_datatype:{LOGITS_DATATYPE}"
        )

    @modal.exit()
    def shutdown(self):
        del self.llm

    def get_config_fingerprint(self, tensorrt_llm_version, config_kwargs):
        """Hash config kwargs so we can cache the built engine."""
        return (
            tensorrt_llm_version
            + "-"
            + self.model
            + "-"
            + hashlib.md5(
                json.dumps(config_kwargs, sort_keys=True).encode()
            ).hexdigest()
        )

    @modal.web_server(port=9000, startup_timeout=STARTUP_TIMEOUT)
    def start(self):
        subprocess.Popen(
            [
                "/usr/bin/python",
                "/opt/tritonserver/python/openai/openai_frontend/main.py",
                "--model-repository",
                "/triton_model_repo",
                "--tokenizer",
                "meta-llama/Llama-3.1-8B-Instruct",
            ]
        )


@tensorrt_llm_cls()
class TensorRTLLMWithTriton(TensorRTLLMWithTritonBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@tensorrt_llm_cls(gpu="H100!:2")
class TensorRTLLMWithTriton_2xH100(TensorRTLLMWithTritonBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@tensorrt_llm_cls(gpu="H100!:4")
class TensorRTLLMWithTriton_4xH100(TensorRTLLMWithTritonBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@tensorrt_llm_cls(gpu="H100!:8", cpu=8)
class TensorRTLLMWithTriton_8xH100(TensorRTLLMWithTritonBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@contextlib.contextmanager
def tensorrt_llm(
    model: str,
    gpu: str,
    region: str,
    server_config: Optional[Mapping[str, Any]] = None,
    profile: bool = False,
):
    import requests

    all_tensorrt_llm_classes = {
        "H100": TensorRTLLMWithTriton,
        "H100:2": TensorRTLLMWithTriton_2xH100,
        "H100:4": TensorRTLLMWithTriton_4xH100,
        "H100:8": TensorRTLLMWithTriton_8xH100,
    }

    if profile:
        raise ValueError("Profiling is not supported for TensorRT-LLM")

    warnings.warn(
        "Region selection is not yet supported for TensorRT-LLM. Spinning up an instance in us-chicago-1..."
    )

    extra_query = {
        "model": model,
        # Sort keys to ensure that this parameter doesn't change between runs
        # with the same vLLM configuration
        "server_config": json.dumps(server_config, sort_keys=True),
        "caller_id": modal.current_function_call_id(),
    }

    try:
        cls = all_tensorrt_llm_classes[gpu]
    except KeyError:
        raise ValueError(f"Invalid GPU: {gpu}")

    url = cls(model="").start.web_url

    # Wait for TensorRT-LLM server to start
    response_code = -1
    print(f"Requesting health at {url}/health with params {extra_query}")

    while response_code != 200:
        response = requests.get(f"{url}/v1/models", params=extra_query)
        response_code = response.status_code
        time.sleep(5)

    print("Connected to TensorRT-LLM instance")
    yield (url, extra_query)
