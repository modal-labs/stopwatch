import contextlib
import os
import subprocess
import time
import urllib.parse
import urllib.request

import modal

from .benchmark import BenchmarkDefaults
from .resources import app, hf_secret, traces_volume


SCALEDOWN_WINDOW = 2 * 60  # 2 minutes
STARTUP_TIMEOUT = 30 * 60  # 30 minutes
TIMEOUT = 60 * 60  # 1 hour
TRACES_PATH = "/traces"
VLLM_PORT = 8000


def vllm_image_factory(docker_tag: str = "v0.7.3"):
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
    gpu="H100!",
    volumes={TRACES_PATH: traces_volume},
    cpu=4,
    memory=65536,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT,
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
            max_containers=1,
            allow_concurrent_inputs=1000,  # Set to a high number to prevent auto-scaling
            scaledown_window=scaledown_window,
            timeout=timeout,
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


@vllm_cls()
class vLLM(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    extra_vllm_args: str = modal.parameter(default="")
    vllm_env_vars: str = modal.parameter(default="")


@vllm_cls(region="us-east-1")
class vLLM_AWS_USEAST1(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    extra_vllm_args: str = modal.parameter(default="")
    vllm_env_vars: str = modal.parameter(default="")


@vllm_cls(region="us-east4")
class vLLM_GCP_USEAST4(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    extra_vllm_args: str = modal.parameter(default="")
    vllm_env_vars: str = modal.parameter(default="")


@vllm_cls(gpu="A100-40GB")
class vLLM_A100_40GB(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    extra_vllm_args: str = modal.parameter(default="")
    vllm_env_vars: str = modal.parameter(default="")


@vllm_cls(gpu="A100-80GB", region="us-east4")
class vLLM_A100_80GB(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    extra_vllm_args: str = modal.parameter(default="")
    vllm_env_vars: str = modal.parameter(default="")


@vllm_cls(gpu="H100!:2", region="us-east4")
class vLLM_2xH100(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    extra_vllm_args: str = modal.parameter(default="")
    vllm_env_vars: str = modal.parameter(default="")


@vllm_cls(gpu="H100!:4", region="us-east4")
class vLLM_4xH100(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    extra_vllm_args: str = modal.parameter(default="")
    vllm_env_vars: str = modal.parameter(default="")


@vllm_cls(image=vllm_image_factory("v0.6.6"))
class vLLM_v0_6_6(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    extra_vllm_args: str = modal.parameter(default="")
    vllm_env_vars: str = modal.parameter(default="")


all_vllm_classes = {
    "v0.7.3": {
        "H100": {
            "us-ashburn-1": vLLM,
            "us-east-1": vLLM_AWS_USEAST1,
            "us-east4": vLLM_GCP_USEAST4,
        },
        "A100-40GB": {
            "us-ashburn-1": vLLM_A100_40GB,
        },
        "A100-80GB": {
            "us-east4": vLLM_A100_80GB,
        },
        "H100:2": {
            "us-east4": vLLM_2xH100,
        },
        "H100:4": {
            "us-east4": vLLM_4xH100,
        },
    },
    "v0.6.6": {
        "H100!": {
            "us-ashburn-1": vLLM_v0_6_6,
        },
    },
}


@contextlib.contextmanager
def vllm(
    docker_tag: str = "v0.7.3",
    extra_query: dict = {},
    gpu: str = BenchmarkDefaults.GPU,
    region: str = BenchmarkDefaults.REGION,
    profile: bool = False,
):
    # Pick vLLM server class
    try:
        cls = all_vllm_classes[docker_tag][gpu.replace("!", "")][region]
    except KeyError:
        raise ValueError(f"Unsupported vLLM configuration: {docker_tag} {gpu} {region}")

    args = urllib.parse.urlencode(extra_query)
    url = cls(model="").start.web_url

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
