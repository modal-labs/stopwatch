import json
import os
import subprocess

import modal

from .constants import MINUTES, VersionDefaults
from .resources import app, hf_cache_volume, hf_secret, traces_volume


HF_CACHE_PATH = "/cache"
PORT = 30000
TRACES_PATH = "/traces"


def sglang_image_factory(docker_tag: str = VersionDefaults.SGLANG):
    return (
        modal.Image.from_registry(
            f"lmsysorg/sglang:{docker_tag}",
            setup_dockerfile_commands=[
                "RUN ln -s /usr/bin/python3 /usr/bin/python",
            ],
        )
        .pip_install(
            "hf-transfer",
            "grpclib",
            "requests",
            # vLLM is needed for its AWQ marlin kernel, but v0.8.5 has a breaking
            # change that makes it incompatible with SGLang's model loader when running
            # some models, e.g. Qwen/Qwen3-235B-A22B.
            "vllm==0.8.4",
        )
        .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .dockerfile_commands(
            [
                "RUN echo '{%- for message in messages %}{{- message.content }}{%- endfor %}' > /home/no-system-prompt.jinja",
                "ENTRYPOINT []",
            ]
        )
        .add_local_python_source("cli")
    )


def sglang_cls(
    image=sglang_image_factory(),
    secrets=[hf_secret],
    gpu="H100!",
    volumes={HF_CACHE_PATH: hf_cache_volume, TRACES_PATH: traces_volume},
    cpu=4,
    memory=4 * 1024,
    scaledown_window=2 * MINUTES,
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


class SGLangBase:
    """A Modal class that runs an SGLang server."""

    @modal.web_server(port=PORT, startup_timeout=30 * MINUTES)
    def start(self):
        """Start an SGLang server."""

        assert self.model, "model must be set, e.g. 'meta-llama/Llama-3.1-8B-Instruct'"
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
                ]
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


@sglang_cls(region="asia-southeast1")
class SGLang_H100_GCP_ASIASOUTHEAST1(SGLangBase):
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
            "asia-southeast1": SGLang_H100_GCP_ASIASOUTHEAST1,
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
    }
}
