import json
import os
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone

import modal

from .constants import MINUTES, VersionDefaults
from .resources import app, hf_cache_volume, hf_secret, startup_metrics_dict

HF_CACHE_PATH = "/cache"
PORT = 30000


def sglang_image_factory(docker_tag: str = VersionDefaults.SGLANG) -> modal.Image:
    """
    Create a Modal image for running an SGLang server.

    :param: docker_tag: The tag of the SGLang Docker image to use.
    :return: A Modal image for running a SGLang server.
    """

    return (
        modal.Image.from_registry(f"lmsysorg/sglang:{docker_tag}")
        .pip_install(
            "hf-transfer",
            "grpclib",
            "requests",
            # vLLM is needed for its AWQ marlin kernel
            "vllm",
        )
        .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .dockerfile_commands(
            [
                "RUN echo '{%- for message in messages %}{{- message.content }}"
                "{%- endfor %}' > /home/no-system-prompt.jinja",
                "ENTRYPOINT []",
            ],
        )
        .add_local_python_source("cli")
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


@sglang_cls(gpu="A10", region="us-ashburn-1")
class SGLang_A10(SGLangBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@sglang_cls(gpu="A10:4", region="us-ashburn-1")
class SGLang_4xA10(SGLangBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@sglang_cls()
class SGLang_H100(SGLangBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@sglang_cls(gpu="H100!:8", cpu=32, memory=64 * 1024)
class SGLang_8xH100(SGLangBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@sglang_cls(gpu="L40S", region="us-ashburn-1", cpu=8, memory=8 * 1024)
class SGLang_L40S(SGLangBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@sglang_cls(gpu="L40S:4", region="us-ashburn-1", cpu=8, memory=8 * 1024)
class SGLang_4xL40S(SGLangBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


sglang_classes = {
    VersionDefaults.SGLANG: {
        "A10": {
            "us-ashburn-1": SGLang_A10,
        },
        "A10:4": {
            "us-ashburn-1": SGLang_4xA10,
        },
        "H100": {
            "us-chicago-1": SGLang_H100,
        },
        "H100:8": {
            "us-chicago-1": SGLang_8xH100,
        },
        "L40S": {
            "us-ashburn-1": SGLang_L40S,
        },
        "L40S:4": {
            "us-ashburn-1": SGLang_4xL40S,
        },
    },
}
