from typing import Any, Mapping, Optional
import contextlib
import json
import subprocess
import time
import urllib.request
import uuid

import modal

from .resources import app, hf_cache_volume, hf_secret, traces_volume


HF_CACHE_PATH = "/cache"
SCALEDOWN_WINDOW = 30  # 30 seconds
SGLANG_PORT = 30000
STARTUP_TIMEOUT = 30 * 60  # 30 minutes
TIMEOUT = 60 * 60  # 1 hour
TRACES_PATH = "/traces"


def sglang_image_factory():
    return (
        modal.Image.from_registry(
            "lmsysorg/sglang",
            setup_dockerfile_commands=[
                "RUN ln -s /usr/bin/python3 /usr/bin/python",
            ],
        )
        .pip_install("hf-transfer", "grpclib", "numpy", "packaging", "SQLAlchemy")
        .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .dockerfile_commands("ENTRYPOINT []")
    )


def sglang_cls(
    image=sglang_image_factory(),
    secrets=[hf_secret],
    gpu="H100!",
    volumes={HF_CACHE_PATH: hf_cache_volume, TRACES_PATH: traces_volume},
    cpu=4,
    memory=65536,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT,
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
            allow_concurrent_inputs=1000,  # Set to a high number to prevent auto-scaling
            scaledown_window=scaledown_window,
            timeout=timeout,
            region=region,
        )(cls)

    return decorator


class SGLangBase:
    """A Modal class that runs an SGLang server."""

    @modal.web_server(port=SGLANG_PORT, startup_timeout=STARTUP_TIMEOUT)
    def start(self):
        """Start an SGLang server."""

        assert self.model, "model must be set, e.g. 'meta-llama/Llama-3.1-8B-Instruct'"
        server_config = json.loads(self.server_config)

        # Start SGLang server
        subprocess.Popen(
            [
                "python",
                "-m",
                "sglang.launch_server",
                "--model-path",
                self.model,
                "--host",
                "0.0.0.0",
                *server_config.get("extra_args", []),
            ]
        )


@sglang_cls()
class SGLang(SGLangBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@sglang_cls(gpu="H100!:2")
class SGLang_2xH100(SGLangBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


all_sglang_classes = {
    "H100": SGLang,
    "H100:2": SGLang_2xH100,
}


@contextlib.contextmanager
def sglang(
    model: str,
    gpu: str,
    region: str,
    server_config: Optional[Mapping[str, Any]] = None,
    profile: bool = False,
):
    # If the SGLang server takes more than 12.5 minutes to start, the metrics
    # endpoint will fail with a 303 infinite redirect error. As a workaround,
    # to handle this issue, we update the caller ID and try to spin up a new
    # SGLang server instead.
    connected = False

    extra_query = {
        "model": model,
        # Sort keys to ensure that this parameter doesn't change between runs
        # with the same SGLang configuration
        "server_config": json.dumps(server_config, sort_keys=True),
        "caller_id": modal.current_function_call_id(),
    }

    while not connected:
        try:
            cls = all_sglang_classes[gpu]
        except KeyError:
            raise ValueError(f"Unsupported SGLang configuration: {gpu}")

        args = urllib.parse.urlencode(extra_query)
        url = cls(model="").start.web_url

        # Wait for SGLang server to start
        print(f"Requesting health check at {url}/health_generate?{args}")

        while True:
            try:
                response = urllib.request.urlopen(f"{url}/health_generate?{args}")
            except urllib.error.HTTPError as e:
                print(f"Error requesting health check: {e}")
                extra_query["caller_id"] = str(uuid.uuid4())
                break

            if response.status == 200:
                connected = True
                break
            else:
                time.sleep(5)

    print("Connected to SGLang instance")
    yield (url, extra_query)
