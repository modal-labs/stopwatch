import contextlib
import os
import subprocess
import time
import urllib.request

import modal

from .resources import app, hf_secret, traces_volume, tunnel_urls

CONTAINER_IDLE_TIMEOUT = 30  # 30 seconds
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

    @modal.method()
    def start(self, caller_id: str, env_vars: dict = {}, vllm_args: list = []):
        """Start the vLLM server.

        Args:
            caller_id (str): ID of the function call that started the vLLM server.
            env_vars (dict): Environment variables to set for the vLLM server.
            vllm_args (list): Arguments to pass to the vLLM server.
        """

        self.caller_id = caller_id

        # Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = value

        with modal.forward(VLLM_PORT) as tunnel:
            print(f"Starting vLLM server at {tunnel.url}")

            # Save tunnel URL so that the benchmarking runner can access it
            tunnel_urls[caller_id] = tunnel.url

            # Start vLLM server
            subprocess.run(
                [
                    "python3.vllm",
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    *vllm_args,
                ]
            )

    @modal.exit()
    def shutdown(self):
        # Remove tunnel URL from dict
        if hasattr(self, "caller_id") and self.caller_id in tunnel_urls:
            tunnel_urls.pop(self.caller_id)

        # Commit traces volume
        traces_volume.commit()

        # Kill vLLM server
        subprocess.run(["pkill", "-9", "python3.vllm"])


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
            container_idle_timeout=container_idle_timeout,
            timeout=timeout,
            cloud=cloud,
            region=region,
        )(cls)

    return decorator


@vllm_cls()
class vLLM(vLLMBase):
    pass


@vllm_cls(image=vllm_image_factory("v0.6.6"))
class vLLM_v0_6_6(vLLMBase):
    pass


@contextlib.contextmanager
def vllm(
    model: str,
    docker_tag: str = "latest",
    env_vars: dict = {},
    extra_args: list = [],
    gpu: str = "H100",
    profile: bool = False,
):
    # Pick vLLM server class
    if docker_tag == "latest":
        cls = vLLM
    elif docker_tag == "v0.6.6":
        cls = vLLM_v0_6_6
    else:
        raise ValueError(f"Invalid vLLM docker tag: {docker_tag}")

    caller_id = modal.current_function_call_id()

    # Start the vLLM server
    vllm = cls.with_options(gpu=gpu)()
    vllm_fc = vllm.start.spawn(
        caller_id=caller_id,
        env_vars=env_vars,
        vllm_args=["--model", model, *extra_args],
    )

    # Wait for vLLM server to start
    while True:
        time.sleep(5)
        url = tunnel_urls.get(caller_id, None)

        if url is None:
            continue

        try:
            urllib.request.urlopen(f"{url}/metrics")
            print(f"Connected to vLLM instance at {url}")
            break
        except Exception:
            continue

    if profile:
        req = urllib.request.Request(f"{url}/start_profile", method="POST")
        urllib.request.urlopen(req)

    try:
        yield url
    finally:
        if profile:
            req = urllib.request.Request(f"{url}/stop_profile", method="POST")
            urllib.request.urlopen(req)

        vllm_fc.cancel()
