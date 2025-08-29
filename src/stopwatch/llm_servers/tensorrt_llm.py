import hashlib
import json
import logging
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import modal

from stopwatch.constants import (
    HF_CACHE_PATH,
    HOURS,
    TENSORRT_LLM_CUDA_VERSION,
    LLMServerType,
)
from stopwatch.resources import hf_cache_volume, startup_metrics_dict

LLM_KWARGS_PATH = "llm_kwargs.yaml"
PORT = 8000

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def tensorrt_llm_image_factory(
    tensorrt_llm_version: str = LLMServerType.tensorrt_llm.get_version(),
    cuda_version: str = TENSORRT_LLM_CUDA_VERSION,
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
        .apt_install("libopenmpi-dev", "git", "git-lfs", "wget")
        .uv_pip_install(
            # This next line is not needed for tensorrt-llm>=1.1.0rc0, but 1.1.0 has a
            # bug when running HF models at the moment. Once 1.1.0 is stable, the line
            # for cuda-python can be removed.
            "cuda-python<13.0",
            "pynvml",
            f"tensorrt-llm=={tensorrt_llm_version}",
            extra_index_url="https://pypi.nvidia.com",
        )
        .uv_pip_install(
            "hf-transfer",
            "huggingface-hub[hf_xet]",
            "requests",
        )
        .env(
            {
                "HF_HUB_CACHE": HF_CACHE_PATH,
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "PMIX_MCA_gds": "hash",
            },
        )
    )


class TensorRTLLMBase:
    """A Modal class that runs a TensorRT-LLM server."""

    @modal.enter()
    def enter(self) -> None:
        """Download the base model and build the TensorRT-LLM engine."""
        import tensorrt_llm
        import yaml

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

            engine_fingerprint = hashlib.md5(  # noqa: S324
                json.dumps(llm_kwargs, sort_keys=True).encode(),
            ).hexdigest()
            logger.info("Engine fingerprint: %s", engine_fingerprint)
            logger.info("%s", llm_kwargs)

            self.config_path = (
                Path(HF_CACHE_PATH)
                / "tensorrt-llm-configs"
                / f"{tensorrt_llm.__version__}-{self.model}-{engine_fingerprint}"
                / LLM_KWARGS_PATH
            )
            logger.info("Config path: %s", self.config_path)

            if not self.config_path.exists():
                # Save the config
                if not self.config_path.parent.exists():
                    self.config_path.parent.mkdir(parents=True)

                with self.config_path.open("w") as f:
                    yaml.dump(llm_kwargs, f)
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    @modal.web_server(port=PORT, startup_timeout=1 * HOURS)
    def start(self) -> None:
        """Start a TensorRT-LLM server."""

        # Save the startup time to a dictionary so we can measure cold start duration
        startup_metrics_dict[self.server_id] = datetime.now(timezone.utc).timestamp()

        if not self.model:
            msg = "model must be set, e.g. 'meta-llama/Llama-3.1-8B-Instruct'"
            raise ValueError(msg)

        server_config = json.loads(self.server_config)

        if hasattr(self, "config_path"):
            # Make sure the volume is up-to-date and this container has access to the
            # saved config.
            for _ in range(10):
                if self.config_path.exists():
                    break
                time.sleep(5)
                hf_cache_volume.reload()
        else:
            # self.config_path is only not set if there was an error in enter(). By
            # setting self.config_path to "none", trtllm-serve will fail, as explained
            # in the comment at the start of enter().
            self.config_path = Path("none")

        # Start TensorRT-LLM server
        subprocess.Popen(
            " ".join(
                [
                    "trtllm-serve",
                    self.model,
                    "--host",
                    "0.0.0.0",
                    "--extra_llm_api_options",
                    str(self.config_path),
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
