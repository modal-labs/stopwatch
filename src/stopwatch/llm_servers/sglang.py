import json
import os
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone

import modal

from stopwatch.constants import HF_CACHE_PATH, MINUTES, VersionDefaults
from stopwatch.resources import app, hf_cache_volume, hf_secret, startup_metrics_dict

PORT = 30000


def sglang_image_factory(
    docker_tag: str = VersionDefaults.SGLANG,
    extra_python_packages: list[str] | None = None,
    transformers_version: str | None = None,
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
            *(
                [f"transformers=={transformers_version}"]
                if transformers_version
                else []
            ),
        )
        .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .dockerfile_commands(
            [
                "RUN echo '{%- for message in messages %}{{- message.content }}"
                "{%- endfor %}' > /home/no-system-prompt.jinja",
                "ENTRYPOINT []",
            ],
        )
    )


def sglang_cls(
    image: modal.Image = sglang_image_factory(),  # noqa: B008
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
    Create an SGLang server class that runs on Modal.

    :param: image: Image to use for the SGLang server.
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
    :return: An SGLang server class that runs on Modal.
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


class SGLangBase:
    """A Modal class that runs an SGLang server."""

    @modal.web_server(port=PORT, startup_timeout=30 * MINUTES)
    def start(self) -> None:
        """Start an SGLang server."""

        # Save the startup time to a dictionary so we can measure cold start duration
        startup_metrics_dict[self.caller_id] = datetime.now(timezone.utc).timestamp()

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
