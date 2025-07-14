import json
import os
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime

import modal

from stopwatch.constants import HOURS, SECONDS, VersionDefaults
from stopwatch.resources import (
    app,
    hf_cache_volume,
    hf_secret,
    startup_metrics_dict,
    traces_volume,
    vllm_cache_volume,
)

HF_CACHE_PATH = "/cache"
PORT = 8000
TRACES_PATH = "/traces"
VLLM_CACHE_PATH = "/root/.cache/vllm"
VLLM_PYTHON_BINARY = "/usr/bin/python3"


def vllm_image_factory(
    docker_tag: str = VersionDefaults.VLLM,
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
        .pip_install("hf-transfer", "grpclib", "requests")
        .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .dockerfile_commands(
            [
                "RUN echo '{%- for message in messages %}{{- message.content }}"
                "{%- endfor %}' > /home/no-system-prompt.jinja",
                *(extra_dockerfile_commands or []),
                "ENTRYPOINT []",
            ],
        )
        .add_local_python_source("cli")
    )


def vllm_cls(
    image: modal.Image = vllm_image_factory(),  # noqa: B008
    secrets: list[modal.Secret] = [hf_secret],  # noqa: B006
    gpu: str = "H100!",
    volumes: dict[str, modal.Volume] = {  # noqa: B006
        HF_CACHE_PATH: hf_cache_volume,
        TRACES_PATH: traces_volume,
        VLLM_CACHE_PATH: vllm_cache_volume,
    },
    cpu: int = 4,
    memory: int = 4 * 1024,
    scaledown_window: int = 30 * SECONDS,
    timeout: int = 1 * HOURS,
    region: str = "us-chicago-1",
) -> Callable:
    """
    Create a vLLM server class that runs on Modal.

    :param: image: Image to use for the vLLM server.
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
    :return: A vLLM server class that runs on Modal.
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


class vLLMBase:
    """A Modal class that runs a vLLM server."""

    @modal.web_server(port=PORT, startup_timeout=1 * HOURS)
    def start(self) -> None:
        """Start a vLLM server."""

        # Save the startup time to a dictionary so we can measure cold start duration
        startup_metrics_dict[self.caller_id] = datetime.now(UTC).timestamp()

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


@vllm_cls(gpu="A10", region="us-ashburn-1")
class vLLM_A10(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls(gpu="A10:4", region="us-ashburn-1")
class vLLM_4xA10(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls(gpu="B200:8", region="us-ashburn-1", cpu=32, memory=64 * 1024)
class vLLM_8xB200(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls()
class vLLM_H100(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls(gpu="H100!:8", cpu=32, memory=64 * 1024)
class vLLM_8xH100(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls(gpu="H200:8", region="us-east-1", cpu=32, memory=64 * 1024)
class vLLM_8xH200(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls(gpu="L40S", region="us-ashburn-1", cpu=8, memory=8 * 1024)
class vLLM_L40S(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls(gpu="L40S:4", region="us-ashburn-1", cpu=8, memory=8 * 1024)
class vLLM_4xL40S(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


vllm_classes = {
    VersionDefaults.VLLM: {
        "A10": {
            "us-ashburn-1": vLLM_A10,
        },
        "A10:4": {
            "us-ashburn-1": vLLM_4xA10,
        },
        "B200:8": {
            "us-ashburn-1": vLLM_8xB200,
        },
        "H100": {
            "us-chicago-1": vLLM_H100,
        },
        "H100:8": {
            "us-chicago-1": vLLM_8xH100,
        },
        "H200:8": {
            "us-east-1": vLLM_8xH200,
        },
        "L40S": {
            "us-ashburn-1": vLLM_L40S,
        },
        "L40S:4": {
            "us-ashburn-1": vLLM_4xL40S,
        },
    },
}
