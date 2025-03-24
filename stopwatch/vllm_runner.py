from typing import Any, Mapping, Optional
import contextlib
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
import uuid

from packaging.version import Version
import modal

from .db import DEFAULT_LLM_SERVER_CONFIGS
from .resources import app, hf_cache_volume, hf_secret, traces_volume


HF_CACHE_PATH = "/cache"
SCALEDOWN_WINDOW = 2 * 60  # 2 minutes
STARTUP_TIMEOUT = 30 * 60  # 30 minutes
TIMEOUT = 60 * 60  # 1 hour
TRACES_PATH = "/traces"
VLLM_PORT = 8000


def vllm_image_factory(docker_tag: str = "v0.7.3"):
    python_binary = (
        "/opt/venv/bin/python3"
        if Version(docker_tag) >= Version("v0.8.0")
        else "/usr/bin/python3"
    )

    return (
        modal.Image.from_registry(
            f"vllm/vllm-openai:{docker_tag}",
            setup_dockerfile_commands=[
                f"RUN echo -n {python_binary} > /vllm-workspace/vllm-python",
            ],
            add_python="3.13",
        )
        .pip_install("hf-transfer", "grpclib", "numpy", "packaging", "SQLAlchemy")
        .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .dockerfile_commands("ENTRYPOINT []")
    )


def vllm_cls(
    image=vllm_image_factory(),
    secrets=[hf_secret],
    gpu="H100!",
    volumes={HF_CACHE_PATH: hf_cache_volume, TRACES_PATH: traces_volume},
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
        server_config = json.loads(self.server_config)

        # Start vLLM server
        subprocess.Popen(
            [
                # Read the location of the correct Python binary from the file
                # created while building the image.
                open("/vllm-workspace/vllm-python").read(),
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                self.model,
                *server_config.get("extra_args", []),
            ],
            env={
                **os.environ,
                **server_config.get("env_vars", {}),
            },
        )


@vllm_cls(image=vllm_image_factory("v0.8.1"), region="us-chicago-1")
class vLLM_v0_8_1(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls(image=vllm_image_factory("v0.8.0"), region="us-chicago-1")
class vLLM_v0_8_0(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls()
class vLLM_OCI_USASHBURN1(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls(region="us-east-1")
class vLLM_AWS_USEAST1(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls(region="us-east4")
class vLLM_GCP_USEAST4(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls(region="us-chicago-1")
class vLLM_OCI_USCHICAGO1(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls(gpu="A100-40GB", region="us-chicago-1")
class vLLM_A100_40GB(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls(gpu="A100-80GB", region="us-east4")
class vLLM_A100_80GB(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls(gpu="H100!:2", region="us-chicago-1")
class vLLM_2xH100(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls(gpu="H100!:4", region="us-chicago-1")
class vLLM_4xH100(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@vllm_cls(image=vllm_image_factory("v0.6.6"))
class vLLM_v0_6_6(vLLMBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


all_vllm_classes = {
    "v0.8.1": {
        "H100": {
            "us-chicago-1": vLLM_v0_8_1,
        },
    },
    "v0.8.0": {
        "H100": {
            "us-chicago-1": vLLM_v0_8_0,
        },
    },
    "v0.7.3": {
        "H100": {
            "us-ashburn-1": vLLM_OCI_USASHBURN1,
            "us-east-1": vLLM_AWS_USEAST1,
            "us-east4": vLLM_GCP_USEAST4,
            "us-chicago-1": vLLM_OCI_USCHICAGO1,
        },
        "A100-40GB": {
            "us-chicago-1": vLLM_A100_40GB,
        },
        "A100-80GB": {
            "us-east4": vLLM_A100_80GB,
        },
        "H100:2": {
            "us-chicago-1": vLLM_2xH100,
        },
        "H100:4": {
            "us-chicago-1": vLLM_4xH100,
        },
    },
    "v0.6.6": {
        "H100": {
            "us-ashburn-1": vLLM_v0_6_6,
        },
    },
}


@contextlib.contextmanager
def vllm(
    model: str,
    gpu: str,
    region: str,
    server_config: Optional[Mapping[str, Any]] = None,
    profile: bool = False,
):
    # If the vLLM server takes more than 12.5 minutes to start, the metrics
    # endpoint will fail with a 303 infinite redirect error. As a workaround,
    # to handle this issue, we update the caller ID and try to spin up a new
    # vLLM server instead.
    connected = False

    docker_tag = server_config.get(
        "docker_tag", DEFAULT_LLM_SERVER_CONFIGS["vllm"]["docker_tag"]
    )

    extra_query = {
        "model": model,
        # Sort keys to ensure that this parameter doesn't change between runs
        # with the same vLLM configuration
        "server_config": json.dumps(server_config, sort_keys=True),
        "caller_id": modal.current_function_call_id(),
    }

    while not connected:
        # Pick vLLM server class
        try:
            cls = all_vllm_classes[docker_tag][gpu.replace("!", "")][region]
        except KeyError:
            raise ValueError(
                f"Unsupported vLLM configuration: {docker_tag} {gpu} {region}"
            )

        args = urllib.parse.urlencode(extra_query)
        url = cls(model="").start.web_url

        # Wait for vLLM server to start
        response_body = ""
        print(f"Requesting metrics at {url}/metrics?{args}")

        while True:
            try:
                response = urllib.request.urlopen(f"{url}/metrics?{args}")
            except urllib.error.HTTPError as e:
                print(f"Error requesting metrics: {e}")
                extra_query["caller_id"] = str(uuid.uuid4())
                break

            response_body = response.read().decode("utf-8")

            if "vllm:gpu_cache_usage_perc" in response_body:
                connected = True
                break
            else:
                time.sleep(5)

    print("Connected to vLLM instance")

    if profile:
        req = urllib.request.Request(f"{url}/start_profile?{args}", method="POST")
        urllib.request.urlopen(req)

    try:
        yield (url, extra_query)
    finally:
        if profile:
            req = urllib.request.Request(f"{url}/stop_profile?{args}", method="POST")
            urllib.request.urlopen(req)
