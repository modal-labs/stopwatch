from typing import Any, Mapping, Optional
import contextlib
import hashlib
import json
import os
import subprocess
import time

import modal

from .constants import HOURS, MINUTES, SECONDS, VersionDefaults
from .resources import app, hf_cache_volume, hf_secret


HF_CACHE_PATH = "/cache"
LLM_KWARGS_PATH = "llm_kwargs.yaml"


def tensorrt_llm_image_factory(
    tensorrt_llm_version: str = VersionDefaults.TENSORRT_LLM,
    cuda_version: str = "12.8.1",
):
    return (
        modal.Image.from_registry(
            f"nvidia/cuda:{cuda_version}-devel-ubuntu24.04",
            add_python="3.12",
        )
        .entrypoint([])  # Remove verbose logging by base image on entry
        .apt_install("openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget")
        .pip_install(
            f"tensorrt-llm=={tensorrt_llm_version}",
            "pynvml",
            extra_index_url="https://pypi.nvidia.com",
        )
        .pip_install(
            "hf-transfer",
            "huggingface_hub",
            "requests",
        )
        .env(
            {
                "HF_HUB_CACHE": HF_CACHE_PATH,
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            }
        )
        .add_local_python_source("cli")
    )


def tensorrt_llm_cls(
    image=tensorrt_llm_image_factory(),
    secrets=[hf_secret],
    gpu="H100!",
    volumes={HF_CACHE_PATH: hf_cache_volume},
    cpu=4,
    memory=65536,
    scaledown_window=30 * SECONDS,
    timeout=1 * HOURS,
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
            scaledown_window=scaledown_window,
            timeout=timeout,
            region=region,
        )(modal.concurrent(max_inputs=1000)(cls))

    return decorator


class TensorRTLLMBase:
    """
    A Modal class that runs an OpenAI-compatible TensorRT-LLM server.
    """

    @modal.enter()
    def enter(self):
        from huggingface_hub import snapshot_download
        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi.llm_args import update_llm_args_with_extra_dict
        from tensorrt_llm.plugin import PluginConfig
        import tensorrt_llm
        import yaml

        server_config = json.loads(self.server_config)
        llm_kwargs = server_config.get("llm_kwargs", {})

        print("Received server config")
        print(server_config)

        print("Downloading base model if necessary")
        model_path = snapshot_download(self.model)

        engine_fingerprint = hashlib.md5(
            json.dumps(llm_kwargs, sort_keys=True).encode()
        ).hexdigest()
        print(f"Engine fingerprint: {engine_fingerprint}")
        print(llm_kwargs)

        self.engine_path = os.path.join(
            model_path,
            "tensorrt-llm-engines",
            f"{tensorrt_llm.__version__}-{self.model}-{engine_fingerprint}",
        )

        if not os.path.exists(self.engine_path):
            # Build the engine
            llm_kwargs_simple = llm_kwargs.copy()

            # Build with plugins, but don't save them to the engine kwargs yaml
            # file, because trtllm-serve doesn't support loading them. This is
            # fine, since the plugins are incorporated at build time.
            if (
                "build_config" in llm_kwargs
                and "plugin_config" in llm_kwargs["build_config"]
            ):
                llm_kwargs["build_config"]["plugin_config"] = PluginConfig.from_dict(
                    llm_kwargs["build_config"]["plugin_config"]
                )
                llm_kwargs_simple["build_config"].pop("plugin_config")

            # Prepare kwargs for LLM constructor
            llm_kwargs = update_llm_args_with_extra_dict(
                {
                    "model": model_path,
                    "tokenizer": server_config.get("tokenizer", self.model),
                },
                llm_kwargs,
            )

            print(f"Building new engine at {self.engine_path}")
            llm = LLM(**llm_kwargs)
            llm.save(self.engine_path)
            del llm

            with open(os.path.join(self.engine_path, LLM_KWARGS_PATH), "w") as f:
                yaml.dump(llm_kwargs_simple, f)

    @modal.web_server(port=8000, startup_timeout=30 * MINUTES)
    def start(self):
        """Start a TensorRT-LLM server."""

        assert self.model, "model must be set, e.g. 'meta-llama/Llama-3.1-8B-Instruct'"
        server_config = json.loads(self.server_config)

        # Start TensorRT-LLM server
        subprocess.Popen(
            [
                "trtllm-serve",
                self.engine_path,
                "--host",
                "0.0.0.0",
                "--extra_llm_api_options",
                os.path.join(self.engine_path, LLM_KWARGS_PATH),
                *(
                    ["--tokenizer", server_config["tokenizer"]]
                    if "tokenizer" in server_config
                    else []
                ),
                *server_config.get("extra_args", []),
            ],
            env={
                **os.environ,
                **server_config.get("env_vars", {}),
            },
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

    if profile:
        raise ValueError("Profiling is not supported for TensorRT-LLM")

    all_tensorrt_llm_classes = {
        VersionDefaults.TENSORRT_LLM: {
            "H100": {
                "us-chicago-1": TensorRTLLM,
            },
            "H100:2": {
                "us-chicago-1": TensorRTLLM_2xH100,
            },
            "H100:4": {
                "us-chicago-1": TensorRTLLM_4xH100,
            },
            "H100:8": {
                "us-chicago-1": TensorRTLLM_8xH100,
            },
        }
    }

    tensorrt_llm_version = server_config.get("version", VersionDefaults.TENSORRT_LLM)
    extra_query = {
        "model": model,
        # Sort keys to ensure that this parameter doesn't change between runs
        # with the same vLLM configuration
        "server_config": json.dumps(server_config, sort_keys=True),
        "caller_id": modal.current_function_call_id(),
    }

    # Pick TensorRT-LLM server class
    try:
        cls = all_tensorrt_llm_classes[tensorrt_llm_version][gpu.replace("!", "")][
            region
        ]
    except KeyError:
        raise ValueError(
            f"Unsupported TensorRT-LLM configuration: {tensorrt_llm_version} {gpu} {region}"
        )

    url = cls(model="").start.web_url

    # Wait for TensorRT-LLM server to start
    print(f"Requesting health check at {url}/health with params {extra_query}")

    num_retries = 3
    for retry in range(num_retries):
        res = requests.get(f"{url}/health", params=extra_query)

        if res.status_code == 200:
            break
        else:
            time.sleep(5)

        if retry == num_retries - 1:
            raise ValueError(
                f"Failed to connect to TensorRT-LLM instance: {res.status_code} {res.text}"
            )

    print("Connected to TensorRT-LLM instance")
    yield (url, extra_query)
