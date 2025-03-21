from typing import Any, Mapping, Optional
import contextlib
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request

import modal

from .db import BenchmarkDefaults
from .resources import app, hf_cache_volume, hf_secret


HF_CACHE_PATH = "/cache"
SCALEDOWN_WINDOW = 2 * 60  # 2 minutes
STARTUP_TIMEOUT = 30 * 60  # 30 minutes
TIMEOUT = 60 * 60  # 1 hour
# TRTLLM_PORT = 8000


def trtllm_image_factory(
    trtllm_version: str = "0.17.0.post1", cuda_version: str = "12.8.0"
):
    return (
        modal.Image.from_registry(
            f"nvidia/cuda:{cuda_version}-devel-ubuntu22.04",
            add_python="3.12",  # TRT-LLM requires Python 3.12
        )
        .entrypoint([])  # remove verbose logging by base image on entry
        .apt_install("openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget")
        .pip_install(
            f"tensorrt-llm=={trtllm_version}",
            "pynvml<12",  # avoid breaking change to pynvml version API
            pre=True,
            extra_index_url="https://pypi.nvidia.com",
        )
        .pip_install(
            "hf-transfer==0.1.9",
            "huggingface_hub==0.28.1",
            "fastapi",
            "numpy",
            "SQLAlchemy",
        )
        .env(
            {
                "HF_HUB_CACHE": HF_CACHE_PATH,
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            }
        )
    )


def trtllm_cls(
    image=trtllm_image_factory(),
    secrets=[hf_secret],
    gpu="H100!",
    volumes={HF_CACHE_PATH: hf_cache_volume},
    cpu=4,
    memory=65536,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT,
    region="us-ashburn-1",
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


class trtLLMBase:
    """
    A Modal class that runs an OpenAI-compatible trtLLM server.
    """

    def construct_engine_kwargs(self) -> dict:
        import torch
        import tensorrt_llm
        from tensorrt_llm import BuildConfig
        from tensorrt_llm.llmapi import (
            QuantConfig,
            KvCacheConfig,
            CalibConfig,
            LookaheadDecodingConfig,
        )
        from tensorrt_llm.plugin.plugin import PluginConfig

        print("converting engine kwarg strings into config objects")

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

        engine_kwargs = json.loads(self.extra_llm_args)
        if "tensor_parallel_size" not in engine_kwargs:
            engine_kwargs["tensor_parallel_size"] = torch.cuda.device_count()
        assert engine_kwargs["tensor_parallel_size"] == torch.cuda.device_count()

        self.config_fingerprint = self.get_config_fingerprint(
            tensorrt_llm.__version__, engine_kwargs
        )

        def construct_configs(x):
            for key, value in x.items():
                if isinstance(value, dict):
                    x[key] = construct_configs(value)
                if key in config_name_to_constructor:
                    x[key] = config_name_to_constructor[key](value)
            return x

        engine_kwargs = construct_configs(engine_kwargs)

        self.lookahead_config = engine_kwargs.get("speculative_config")

        print("number of GPUs:", engine_kwargs["tensor_parallel_size"])
        assert engine_kwargs["tensor_parallel_size"] == torch.cuda.device_count()

        return engine_kwargs

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

        self.model_path = os.path.join(HF_CACHE_PATH, self.model)

        self.seed_everything()

        print("downloading base model if necessary")
        hf_cache_volume.reload()
        snapshot_download(self.model, local_dir=self.model_path)
        hf_cache_volume.commit()

        engine_kwargs = self.construct_engine_kwargs()

        # FIXME just make this trttlm version-engine
        engine_path = os.path.join(
            self.model_path, "trtllm_engine", self.config_fingerprint
        )
        if not os.path.exists(engine_path):
            self.build_engine(engine_path, engine_kwargs)

        print(f"loading engine from {engine_path}")
        self.llm = LLM(model=engine_path, **engine_kwargs)

    @modal.asgi_app()
    def start(self):
        from transformers import AutoTokenizer
        from tensorrt_llm.serve import OpenAIServer
        from fastapi.responses import JSONResponse

        """Start a trtLLM server."""

        assert self.model, "model must be set, e.g. 'meta-llama/Llama-3.1-8B-Instruct'"
        assert self.llm_env_vars == "", "not supported yet"

        tokenizer = AutoTokenizer.from_pretrained(self.model)
        server = OpenAIServer(llm=self.llm, model=self.model, hf_tokenizer=tokenizer)

        async def get_iteration_stats(self) -> JSONResponse:
            stats = []
            return JSONResponse(content=stats)
            # async for stat in self.llm.get_stats_async(2):
            # stats.append(stat)
            # return JSONResponse(content=stats)

        server.app.add_api_route("/metrics", get_iteration_stats, methods=["GET"])

        return server.app

    @modal.exit()
    def shutdown(self):
        del self.llm

    def seed_everything(self, seed=0):
        import torch
        import numpy as np
        import random

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_config_fingerprint(self, trtllm_version, config_kwargs):
        """Hash config kwargs so we can cache the built engine."""
        return (
            trtllm_version
            + "-"
            + self.model
            + "-"
            + hashlib.md5(
                json.dumps(config_kwargs, sort_keys=True).encode()
            ).hexdigest()
        )


@trtllm_cls(region="us-chicago-1")
class trtLLM(trtLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    extra_llm_args: str = modal.parameter(default="{}")
    llm_env_vars: str = modal.parameter(default="")


@contextlib.contextmanager
def trtllm(
    llm_server_config: Optional[Mapping[str, Any]] = None,
    extra_query: dict = {},
    gpu: str = "H100!",
    region: str = BenchmarkDefaults.REGION,
    profile: bool = False,
):

    cls = trtLLM

    args = urllib.parse.urlencode(extra_query)
    url = cls(model="", extra_trtllm_args="", trtllm_env_vars="").start.web_url

    # Wait for trtLLM server to start
    response_code = -1
    print(f"Requesting health at {url}/health?{args}")

    while response_code != 200:
        response = urllib.request.urlopen(f"{url}/health?{args}")
        response_code = response.code
        time.sleep(5)

    print("Connected to trtLLM instance")

    try:
        yield url
    finally:
        pass
