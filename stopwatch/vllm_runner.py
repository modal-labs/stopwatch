import contextlib
import os
import subprocess
import urllib.request

import modal

from .resources import app, hf_secret, traces_volume

CONTAINER_IDLE_TIMEOUT = 30  # 30 seconds
STARTUP_TIMEOUT = 5 * 60  # 5 minutes
TIMEOUT = 60 * 60  # 1 hour
TRACES_PATH = "/traces"
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

    env_vars: dict[str, str] = {}
    vllm_args: list[str] = []

    @modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT)
    def start(self):
        """Start a vLLM server."""

        assert self.vllm_args, "vllm_args must be set"

        # Start vLLM server
        subprocess.Popen(
            [
                "python3.vllm",
                "-m",
                "vllm.entrypoints.openai.api_server",
                *self.vllm_args,
            ],
            env={
                **os.environ,
                **self.env_vars,
            },
        )


def vllm_cls(
    image=vllm_image_factory(),
    secrets=[hf_secret],
    gpu=modal.gpu.H100(),
    volumes={TRACES_PATH: traces_volume},
    cpu=4,
    memory=65536,
    container_idle_timeout=CONTAINER_IDLE_TIMEOUT,
    timeout=TIMEOUT,
    cloud="oci",
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
            # Set to a high number to prevent auto-scaling
            allow_concurrent_inputs=1000,
            container_idle_timeout=container_idle_timeout,
            timeout=timeout,
            cloud=cloud,
            region=region,
        )(cls)

    return decorator


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
