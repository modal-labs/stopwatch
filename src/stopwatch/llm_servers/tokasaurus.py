import json
import os
import subprocess
from datetime import datetime, timezone

import modal

from stopwatch.constants import (
    HF_CACHE_PATH,
    MINUTES,
    TOKASAURUS_CUDA_VERSION,
    LLMServerType,
)
from stopwatch.resources import hf_cache_volume, startup_metrics_dict

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
