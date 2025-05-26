import json
import os
import subprocess

import modal

from .constants import HOURS, MINUTES, SECONDS, VersionDefaults
from .resources import app, hf_cache_volume, hf_secret, traces_volume


HF_CACHE_PATH = "/cache"
PORT = 8000
TRACES_PATH = "/traces"


def vllm_image_factory(docker_tag: str = VersionDefaults.VLLM, use_cu128: bool = False):
    python_binary = (
        "/opt/venv/bin/python3"
        if docker_tag in ["v0.8.0", "v0.8.1"]
        else "/usr/bin/python3"
    )

    return (
        modal.Image.from_registry(
            (
                "nvcr.io/nvidia/tritonserver:25.04-vllm-python-py3"
                if use_cu128
                else f"vllm/vllm-openai:{docker_tag}"
            ),
            add_python="3.13",
        )
        .pip_install("hf-transfer", "grpclib", "requests")
        .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .dockerfile_commands(
            [
                f"RUN echo -n {python_binary} > /home/vllm-python",
                "RUN echo '{%- for message in messages %}{{- message.content }}{%- endfor %}' > /home/no-system-prompt.jinja",
                "ENTRYPOINT []",
            ]
        )
        .add_local_python_source("cli")
    )


def vllm_cls(
    image=vllm_image_factory(),
    secrets=[hf_secret],
    gpu="H100!",
    volumes={HF_CACHE_PATH: hf_cache_volume, TRACES_PATH: traces_volume},
    cpu=4,
    memory=4 * 1024,
    scaledown_window=30 * SECONDS,
    timeout=2 * MINUTES,
    region="us-chicago-1",
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
            scaledown_window=scaledown_window,
            timeout=timeout,
            region=region,
        )(modal.concurrent(max_inputs=1000)(cls))

    return decorator


class vLLMBase:
    """
    A Modal class that runs a vLLM server. The endpoint is exposed via a
    tunnel, the URL for which is stored in a shared dict.
    """

    @modal.web_server(port=PORT, startup_timeout=1 * HOURS)
    def start(self):
        """Start a vLLM server."""

        assert self.model, "model must be set, e.g. 'meta-llama/Llama-3.1-8B-Instruct'"
        server_config = json.loads(self.server_config)

        # Read the location of the correct Python binary from the file created while
        # building the image.
        python_binary = open("/home/vllm-python").read()

        # Start vLLM server
        subprocess.Popen(
            " ".join(
                [
                    # Read the location of the correct Python binary from the file
                    # created while building the image.
                    python_binary,
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
                ]
            )
            + f" || {python_binary} -m http.server {PORT}",
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


@vllm_cls(
    image=vllm_image_factory(use_cu128=True),
    gpu="B200:8",
    region="us-ashburn-1",
    cpu=32,
    memory=64 * 1024,
)
class vLLM_8xB200(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls()
class vLLM_H100(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls(region="asia-southeast1")
class vLLM_H100_GCP_ASIASOUTHEAST1(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls(gpu="H100!:8", cpu=32, memory=64 * 1024)
class vLLM_8xH100(vLLMBase):
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
            "asia-southeast1": vLLM_H100_GCP_ASIASOUTHEAST1,
        },
        "H100:8": {
            "us-chicago-1": vLLM_8xH100,
        },
        "L40S": {
            "us-ashburn-1": vLLM_L40S,
        },
        "L40S:4": {
            "us-ashburn-1": vLLM_4xL40S,
        },
    }
}
