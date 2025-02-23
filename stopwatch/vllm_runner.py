import contextlib
import os
import subprocess
import urllib.request

import modal


STARTUP_TIMEOUT = 5 * 60  # 5 minutes
VLLM_PORT = 8000


def vllm_image_factory(docker_tag: str = "latest"):
    return modal.Image.from_registry(
        f"vllm/vllm-openai:{docker_tag}",
        setup_dockerfile_commands=[
            "RUN ln -s /usr/bin/python3 /usr/bin/python3.vllm",
        ],
        add_python="3.12",
    ).dockerfile_commands("ENTRYPOINT []")


class vLLMBase:
    """
    A Modal class that runs a vLLM server. The endpoint is exposed via a
    tunnel, the URL for which is stored in a shared dict.
    """

    extra_vllm_args: list[str] = []
    model: str
    vllm_env_vars: dict[str, str] = {}

    @modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT)
    def start(self):
        """Start a vLLM server."""

        assert self.model, "model must be set, e.g. 'meta-llama/Llama-3.1-8B-Instruct'"

        # Start vLLM server
        subprocess.Popen(
            [
                "python3.vllm",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                self.model,
                *self.extra_vllm_args,
            ],
            env={
                **os.environ,
                **self.vllm_env_vars,
            },
        )


@contextlib.contextmanager
def vllm(vllm_deployment_id: str, profile: bool = False):
    # Start the vLLM server
    url = f"https://jackcook--stopwatch-vllm-{vllm_deployment_id}-start.modal.run"

    # Wait for vLLM server to start
    urllib.request.urlopen(f"{url}/metrics")
    print(f"Connected to vLLM instance at {url}")

    if profile:
        req = urllib.request.Request(f"{url}/start_profile", method="POST")
        urllib.request.urlopen(req)

    try:
        yield url
    finally:
        if profile:
            req = urllib.request.Request(f"{url}/stop_profile", method="POST")
            urllib.request.urlopen(req)
