from typing import Any, Mapping, Optional
import contextlib
import hashlib
import json
import os
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
            f"nvidia/cuda:{cuda_version}-devel-ubuntu22.04",
            add_python="3.12",  # TRT-LLM requires Python 3.12
        )
        .entrypoint([])  # remove verbose logging by base image on entry
        .apt_install("openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget")
        .pip_install(
            f"tensorrt-llm=={tensorrt_llm_version}",
            "pynvml<12",  # avoid breaking change to pynvml version API
            pre=True,
            extra_index_url="https://pypi.nvidia.com",
        )
        .pip_install(
            "hf-transfer==0.1.9",
            "huggingface_hub==0.28.1",
            "fastapi",
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


class TensorRTLLMBase:
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
            "plugin_config": PluginConfig.from_dict,
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
        from tensorrt_llm import LLM

        print("Received server config:")
        print(self.server_config)

        print("Downloading base model if necessary")
        self.model_path = snapshot_download(self.model)

        engine_kwargs = self.construct_engine_kwargs()

        # FIXME just make this trttlm version-engine
        engine_path = os.path.join(
            self.model_path, "tensorrt_llm_engine", self.config_fingerprint
        )
        if not os.path.exists(engine_path):
            self.build_engine(engine_path, engine_kwargs)

        print(f"loading engine from {engine_path}")
        self.llm = LLM(model=engine_path, **engine_kwargs)

    @modal.asgi_app()
    def start(self):
        from transformers import AutoTokenizer
        from tensorrt_llm.serve import OpenAIServer

        """Start a TensorRT-LLM server."""

        assert self.model, "model must be set, e.g. 'meta-llama/Llama-3.1-8B-Instruct'"

        engine_kwargs = self.construct_engine_kwargs()
        tokenizer = AutoTokenizer.from_pretrained(
            engine_kwargs.get("tokenizer", self.model)
        )
        server = OpenAIServer(llm=self.llm, model=self.model, hf_tokenizer=tokenizer)

        return server.app

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


@tensorrt_llm_cls()
class TensorRTLLM(TensorRTLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@tensorrt_llm_cls(gpu="H100!:2")
class TensorRTLLM_2xH100(TensorRTLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@tensorrt_llm_cls(gpu="H100!:4")
class TensorRTLLM_4xH100(TensorRTLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@tensorrt_llm_cls(gpu="H100!:8", cpu=8)
class TensorRTLLM_8xH100(TensorRTLLMBase):
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
        "H100": TensorRTLLM,
        "H100:2": TensorRTLLM_2xH100,
        "H100:4": TensorRTLLM_4xH100,
        "H100:8": TensorRTLLM_8xH100,
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
        response = requests.get(f"{url}/health", params=extra_query)
        response_code = response.status_code
        time.sleep(5)

    print("Connected to TensorRT-LLM instance")
    yield (url, extra_query)
