import json
import os
import subprocess
from datetime import datetime, timezone

import modal

from stopwatch.constants import HF_CACHE_PATH, MINUTES, LLMServerType
from stopwatch.resources import hf_cache_volume, startup_metrics_dict

PORT = 30000


def sglang_image_factory(
    docker_tag: str = LLMServerType.sglang.get_version(),
    extra_python_packages: list[str] | None = None,
) -> modal.Image:
    """
    Create a Modal image for running an SGLang server.

    :param: docker_tag: The tag of the SGLang Docker image to use.
    :return: A Modal image for running a SGLang server.
    """

    return (
        modal.Image.from_registry(f"lmsysorg/sglang:{docker_tag}")
        .uv_pip_install(
            "hf-transfer",
            "grpclib",
            "requests",
            # vLLM is needed for its AWQ marlin kernel
            "vllm",
            *(extra_python_packages or []),
        )
        .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .dockerfile_commands("ENTRYPOINT []")
    )


class SGLangBase:
    """A Modal class that runs an SGLang server."""

    @modal.web_server(port=PORT, startup_timeout=30 * MINUTES)
    def start(self) -> None:
        """Start an SGLang server."""

        # Save the startup time to a dictionary so we can measure cold start duration
        startup_metrics_dict[self.server_id] = datetime.now(timezone.utc).timestamp()

        hf_cache_volume.reload()

        if not self.model:
            msg = "model must be set, e.g. 'meta-llama/Llama-3.1-8B-Instruct'"
            raise ValueError(msg)

        server_config = json.loads(self.server_config)

        # Start SGLang server
        subprocess.Popen(
            " ".join(
                [
                    "python",
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    self.model,
                    "--host",
                    "0.0.0.0",
                    *(
                        ["--tokenizer-path", server_config["tokenizer"]]
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
