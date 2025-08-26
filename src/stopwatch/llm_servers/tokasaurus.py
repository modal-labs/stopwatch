import json
import os
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone

import modal

from stopwatch.constants import (
    HF_CACHE_PATH,
    MINUTES,
    TOKASAURUS_CUDA_VERSION,
    LLMServerType,
)
from stopwatch.resources import app, hf_cache_volume, hf_secret, startup_metrics_dict

PORT = 10210


def tokasaurus_image_factory(
    version: str = LLMServerType.tokasaurus.get_version(),
    cuda_version: str = TOKASAURUS_CUDA_VERSION,
) -> modal.Image:
    """
    Create a Modal image for running a Tokasaurus server.

    :param: version: The version of Tokasaurus to install.
    :param: cuda_version: The version of CUDA to start the image from.
    :return: A Modal image for running a Tokasaurus server.
    """

    return (
        modal.Image.from_registry(
            f"nvidia/cuda:{cuda_version}-devel-ubuntu22.04",
            add_python="3.12",
        )
        .entrypoint([])  # Remove verbose logging by base image on entry
        .apt_install("git")
        .uv_pip_install(f"tokasaurus=={version}")
        .env(
            {
                "HF_HUB_CACHE": HF_CACHE_PATH,
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            },
        )
    )


def tokasaurus_cls(
    image: modal.Image = tokasaurus_image_factory(),  # noqa: B008
    secrets: list[modal.Secret] = [hf_secret],  # noqa: B006
    gpu: str = "H100!",
    volumes: dict[str, modal.Volume] = {HF_CACHE_PATH: hf_cache_volume},  # noqa: B006
    cpu: int = 4,
    memory: int = 4 * 1024,
    scaledown_window: int = 2 * MINUTES,
    timeout: int = 30 * MINUTES,
    region: str = "us-chicago-1",
) -> Callable:
    """
    Create a Tokasaurus server class that runs on Modal.

    :param: image: Image to use for the Tokasaurus server.
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
    :return: A Tokasaurus server class that runs on Modal.
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


class TokasaurusBase:
    """A Modal class that runs a Tokasaurus server."""

    @modal.web_server(port=PORT, startup_timeout=30 * MINUTES)
    def start(self) -> None:
        """Start a Tokasaurus server."""

        # Save the startup time to a dictionary so we can measure cold start duration
        startup_metrics_dict[self.server_id] = datetime.now(timezone.utc).timestamp()

        hf_cache_volume.reload()

        if not self.model:
            msg = "model must be set, e.g. 'meta-llama/Llama-3.1-8B-Instruct'"
            raise ValueError(msg)

        server_config = json.loads(self.server_config)

        # Start Tokasaurus server
        subprocess.Popen(
            " ".join(
                [
                    "toka",
                    f"model={self.model}",
                    *(
                        [f"tokenizer={server_config['tokenizer']}"]
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
