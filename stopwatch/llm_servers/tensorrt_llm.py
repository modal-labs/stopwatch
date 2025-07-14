import hashlib
import json
import logging
import os
import subprocess
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import modal

from stopwatch.constants import MINUTES, SECONDS, VersionDefaults
from stopwatch.resources import app, hf_cache_volume, hf_secret, startup_metrics_dict

HF_CACHE_PATH = "/cache"
LLM_KWARGS_PATH = "llm_kwargs.yaml"
PORT = 8000

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def tensorrt_llm_image_factory(
    tensorrt_llm_version: str = VersionDefaults.TENSORRT_LLM,
    cuda_version: str = "12.8.1",
) -> modal.Image:
    """
    Create a Modal image for running a TensorRT-LLM server.

    :param: tensorrt_llm_version: The version of TensorRT-LLM to install.
    :param: cuda_version: The version of CUDA to start the image from.
    :return: A Modal image for running a TensorRT-LLM server.
    """

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
            "huggingface_hub[hf_xet]",
            "requests",
        )
        .env(
            {
                "HF_HUB_CACHE": HF_CACHE_PATH,
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "PMIX_MCA_gds": "hash",
            },
        )
        .add_local_python_source("cli")
    )


def tensorrt_llm_cls(
    image: modal.Image = tensorrt_llm_image_factory(),  # noqa: B008
    secrets: list[modal.Secret] = [hf_secret],  # noqa: B006
    gpu: str = "H100!",
    volumes: dict[str, modal.Volume] = {HF_CACHE_PATH: hf_cache_volume},  # noqa: B006
    cpu: int = 4,
    memory: int = 4 * 1024,
    scaledown_window: int = 30 * SECONDS,
    timeout: int = 30 * MINUTES,
    region: str = "us-chicago-1",
) -> Callable:
    """
    Create a TensorRT-LLM server class that runs on Modal.

    :param: image: Image to use for the TensorRT-LLM server.
    :param: secrets: Secrets to add to the container.
    :param: gpu: GPU to attach to the server's container.
    :param: volumes: Modal volumes to attach to the server's container.
    :param: cpu: Number of CPUs to add to the server.
    :param: memory: RAM, in MB, to add to the server.
    :param: scaledown_window: Time, in seconds, to wait between requests before scaling
        down the server.
    :param: timeout: Time, in seconds, to wait after startup before scaling down the
        server.
    :param: region: Region in which to run the server.
    :return: A TensorRT-LLM server class that runs on Modal.
    """

    def decorator(cls: type) -> Callable:
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
    """A Modal class that runs a TensorRT-LLM server."""

    @modal.enter()
    def enter(self) -> None:
        """Download the base model and build the TensorRT-LLM engine."""
        import tensorrt_llm
        import torch
        import yaml
        from huggingface_hub import snapshot_download
        from tensorrt_llm._tensorrt_engine import LLM
        from tensorrt_llm.llmapi.llm_args import update_llm_args_with_extra_dict
        from tensorrt_llm.plugin import PluginConfig

        # This entire function needs to be wrapped in a try/except block. If an error
        # occurs here, a crash will result in the container getting automatically
        # restarted indefinitely. By logging the error and returning, the container
        # will fail on the trtllm-serve command and then start the http.server module,
        # which will return a 404 error that will be returned to the client.

        try:
            server_config = json.loads(self.server_config)
            llm_kwargs = server_config.get("llm_kwargs", {})

            logger.info("Received server config")
            logger.info(server_config)

            logger.info("Downloading base model if necessary")
            model_path = snapshot_download(self.model)

            engine_fingerprint = hashlib.md5(  # noqa: S324
                json.dumps(llm_kwargs, sort_keys=True).encode(),
            ).hexdigest()
            logger.info("Engine fingerprint: %s", engine_fingerprint)
            logger.info("%s", llm_kwargs)

            self.engine_path = (
                Path(model_path)
                / "tensorrt-llm-engines"
                / f"{tensorrt_llm.__version__}-{self.model}-{engine_fingerprint}"
            )
            logger.info("Engine path: %s", self.engine_path)

            if not self.engine_path.exists():
                # Build the engine
                llm_kwargs_simple = llm_kwargs.copy()

                # Build with plugins, but don't save them to the engine kwargs yaml
                # file, because trtllm-serve doesn't support loading them. This is
                # fine, since the plugins are incorporated at build time.
                if (
                    "build_config" in llm_kwargs
                    and "plugin_config" in llm_kwargs["build_config"]
                ):
                    llm_kwargs["build_config"]["plugin_config"] = (
                        PluginConfig.from_dict(
                            llm_kwargs["build_config"]["plugin_config"],
                        )
                    )
                    llm_kwargs_simple["build_config"].pop("plugin_config")

                # Prepare kwargs for LLM constructor
                llm_kwargs = update_llm_args_with_extra_dict(
                    {
                        "model": model_path,
                        "tensor_parallel_size": torch.cuda.device_count(),
                        "tokenizer": server_config.get("tokenizer", self.model),
                    },
                    llm_kwargs,
                )

                logger.info("Building new engine at %s", self.engine_path)
                llm = LLM(**llm_kwargs)
                llm.save(self.engine_path)
                del llm

                with (self.engine_path / LLM_KWARGS_PATH).open("w") as f:
                    yaml.dump(llm_kwargs_simple, f)
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    @modal.web_server(port=PORT, startup_timeout=30 * MINUTES)
    def start(self) -> None:
        """Start a TensorRT-LLM server."""

        # Save the startup time to a dictionary so we can measure cold start duration
        startup_metrics_dict[self.caller_id] = datetime.now(UTC).timestamp()

        if not self.model:
            msg = "model must be set, e.g. 'meta-llama/Llama-3.1-8B-Instruct'"
            raise ValueError(msg)

        server_config = json.loads(self.server_config)

        # self.engine_path is only not set if there was an error in enter(). By setting
        # engine_path to "none", trtllm-serve will fail, as explained in the comment
        # at the start of enter().
        engine_path = self.engine_path if hasattr(self, "engine_path") else Path("none")
        engine_config_path = self.engine_path / LLM_KWARGS_PATH

        # Make sure the volume is up-to-date and this container has access to the built
        # engine.
        for _ in range(10):
            if engine_config_path.exists():
                break
            time.sleep(5)
            hf_cache_volume.reload()

        # Start TensorRT-LLM server
        subprocess.Popen(
            " ".join(
                [
                    "trtllm-serve",
                    str(engine_path),
                    "--host",
                    "0.0.0.0",
                    "--extra_llm_api_options",
                    str(engine_config_path),
                    *(
                        ["--tokenizer", server_config["tokenizer"]]
                        if "tokenizer" in server_config
                        else []
                    ),
                    *server_config.get("extra_args", []),
                ],
            )
            + f" || python -m http.server {PORT}",
            env={
                **os.environ,
                **server_config.get("env_vars", {}),
            },
            shell=True,
        )


@tensorrt_llm_cls()
class TensorRTLLM(TensorRTLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@tensorrt_llm_cls(gpu="H100!:8", cpu=32, memory=64 * 1024)
class TensorRTLLM_8xH100(TensorRTLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


tensorrt_llm_classes = {
    VersionDefaults.TENSORRT_LLM: {
        "H100": {
            "us-chicago-1": TensorRTLLM,
        },
        "H100:8": {
            "us-chicago-1": TensorRTLLM_8xH100,
        },
    },
}
