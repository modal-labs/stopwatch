from typing import Optional
import json
import os
import subprocess

import modal

from .constants import MINUTES, VersionDefaults


HF_CACHE_PATH = "/cache"
PORT = 30000


def sglang_image_factory(docker_tag: Optional[str] = None):
    if docker_tag is None:
        docker_tag = VersionDefaults.SGLANG

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
