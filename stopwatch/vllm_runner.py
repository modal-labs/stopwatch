import contextlib
import os
import subprocess
import time
import urllib.parse
import urllib.request

import modal

from .resources import app, hf_secret, traces_volume


CONTAINER_IDLE_TIMEOUT = 5 * 60  # 5 minutes
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
    region="us-ashburn-1",
):
    def decorator(cls):
        return app.cls(
            image=image,
            secrets=secrets,
            gpu=gpu,
            volumes=volumes,
            cpu=cpu,
            memory=memory,
            concurrency_limit=1,
            allow_concurrent_inputs=1000,  # Set to a high number to prevent auto-scaling
            container_idle_timeout=container_idle_timeout,
            timeout=timeout,
            cloud=cloud,
            region=region,
        )(cls)

    return decorator


class vLLMBase:
    """
    A Modal class that runs a vLLM server. The endpoint is exposed via a
    tunnel, the URL for which is stored in a shared dict.
    """

    @modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT)
    def start(self):
        """Start a vLLM server."""

        assert self.model, "model must be set, e.g. 'meta-llama/Llama-3.1-8B-Instruct'"

        extra_vllm_args = (
            self.extra_vllm_args.split(" ") if self.extra_vllm_args else []
        )
        vllm_env_vars = (
            {
                k: v
                for k, v in (pair.split("=") for pair in self.vllm_env_vars.split(" "))
            }
            if self.vllm_env_vars
            else {}
        )

        # Start vLLM server
        subprocess.Popen(
            [
                "python3.vllm",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                self.model,
                *extra_vllm_args,
            ],
            env={
                **os.environ,
                **vllm_env_vars,
            },
        )


@vllm_cls(image=vllm_image_factory("v0.7.3"))
class vLLM_v0_7_3(vLLMBase):
    model: str = modal.parameter()
    extra_vllm_args: str = modal.parameter(default="")
    vllm_env_vars: str = modal.parameter(default="")


@vllm_cls(image=vllm_image_factory("v0.6.6"))
class vLLM_v0_6_6(vLLMBase):
    model: str = modal.parameter()
    extra_vllm_args: str = modal.parameter(default="")
    vllm_env_vars: str = modal.parameter(default="")


@contextlib.contextmanager
def vllm(
    docker_tag: str = "latest",
    extra_query: dict = {},
    gpu: str = "H100",
    profile: bool = False,
):
    # Pick vLLM server class
    if docker_tag == "v0.7.3" and gpu == "H100":
        cls = vLLM_v0_7_3
    elif docker_tag == "v0.6.6" and gpu == "H100":
        cls = vLLM_v0_6_6
    else:
        raise ValueError(f"Unsupported vLLM configuration: {docker_tag} {gpu}")

    args = urllib.parse.urlencode(extra_query)
    url = cls().start.web_url

    # Wait for vLLM server to start
    response_body = ""
    print(f"Requesting metrics at {url}/metrics?{args}")

    while "vllm:num_requests_running" not in response_body:
        response = urllib.request.urlopen(f"{url}/metrics?{args}")
        response_body = response.read().decode("utf-8")
        time.sleep(5)

    print("Connected to vLLM instance")

    if profile:
        req = urllib.request.Request(f"{url}/start_profile?{args}", method="POST")
        urllib.request.urlopen(req)

    try:
        yield url
    finally:
        if profile:
            req = urllib.request.Request(f"{url}/stop_profile?{args}", method="POST")
            urllib.request.urlopen(req)
