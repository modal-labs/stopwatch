from typing import Optional
import json
import os
import subprocess

import modal

from .constants import HOURS, VersionDefaults


HF_CACHE_PATH = "/cache"
PORT = 8000
TRACES_PATH = "/traces"


def vllm_image_factory(docker_tag: Optional[str] = None):
    if docker_tag is None:
        docker_tag = VersionDefaults.VLLM

    python_binary = (
        "/opt/venv/bin/python3"
        if docker_tag in ["v0.8.0", "v0.8.1"]
        else "/usr/bin/python3"
    )

    return (
        modal.Image.from_registry(
            f"vllm/vllm-openai:{docker_tag}",
            setup_dockerfile_commands=[
                f"RUN echo -n {python_binary} > /home/vllm-python",
            ],
            add_python="3.13",
        )
        .pip_install("hf-transfer", "grpclib", "requests")
        .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .dockerfile_commands(
            [
                "RUN echo '{%- for message in messages %}{{- message.content }}{%- endfor %}' > /home/no-system-prompt.jinja",
                "ENTRYPOINT []",
            ]
        )
        .add_local_python_source("cli")
    )


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
