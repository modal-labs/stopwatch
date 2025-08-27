import json
import os
import subprocess
from datetime import datetime, timezone

import modal

from stopwatch.constants import HF_CACHE_PATH, HOURS, LLMServerType
from stopwatch.resources import hf_cache_volume, startup_metrics_dict, vllm_cache_volume

PORT = 8000
VLLM_PYTHON_BINARY = "/usr/bin/python3"


def vllm_image_factory(
    docker_tag: str = LLMServerType.vllm.get_version(),
    extra_dockerfile_commands: list[str] | None = None,
) -> modal.Image:
    """
    Create a Modal image for running a vLLM server.

    :param: docker_tag: The tag of the vLLM Docker image to use.
    :param: extra_dockerfile_commands: Extra Dockerfile commands to add to the image.
    :return: A Modal image for running a vLLM server.
    """

    return (
        modal.Image.from_registry(
            f"vllm/vllm-openai:{docker_tag}",
            add_python="3.13",
        )
        .uv_pip_install("hf-transfer", "grpclib", "requests", "typer")
        .env(
            {
                "HF_HUB_CACHE": HF_CACHE_PATH,
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "VLLM_SKIP_P2P_CHECK": "1",
            },
        )
        .dockerfile_commands(*(extra_dockerfile_commands or []), "ENTRYPOINT []")
    )


class vLLMBase:
    """A Modal class that runs a vLLM server."""

    @modal.web_server(port=PORT, startup_timeout=1 * HOURS)
    def start(self) -> None:
        """Start a vLLM server."""

        # Save the startup time to a dictionary so we can measure cold start duration
        startup_metrics_dict[self.server_id] = datetime.now(timezone.utc).timestamp()

        hf_cache_volume.reload()
        vllm_cache_volume.reload()

        if not self.model:
            msg = "model must be set, e.g. 'meta-llama/Llama-3.1-8B-Instruct'"
            raise ValueError(msg)

        server_config = json.loads(self.server_config)

        # Start vLLM server
        subprocess.Popen(
            " ".join(
                [
                    VLLM_PYTHON_BINARY,
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    self.model,
                    *(
                        ["--tokenizer", server_config["tokenizer"]]
                        if "tokenizer" in server_config
                        else []
                    ),
                    *server_config.get("extra_args", []),
                ],
            )
            + f" || {VLLM_PYTHON_BINARY} -m http.server {PORT}",
            env={
                **os.environ,
                **server_config.get("env_vars", {}),
            },
            shell=True,
        )
