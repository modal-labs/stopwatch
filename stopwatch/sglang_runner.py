from typing import Any, Mapping, Optional
import contextlib
import json
import subprocess
import time

import modal

from .resources import app, hf_cache_volume, hf_secret, traces_volume


DEFAULT_DOCKER_TAG = "v0.4.6.post2-cu124"
HF_CACHE_PATH = "/cache"
SCALEDOWN_WINDOW = 2 * 60  # 2 minutes
SGLANG_PORT = 30000
STARTUP_TIMEOUT = 30 * 60  # 30 minutes
TIMEOUT = 60 * 60  # 1 hour
TRACES_PATH = "/traces"


def sglang_image_factory(docker_tag: str = DEFAULT_DOCKER_TAG):
    return (
        modal.Image.from_registry(
            f"lmsysorg/sglang:{docker_tag}",
            setup_dockerfile_commands=[
                "RUN ln -s /usr/bin/python3 /usr/bin/python",
            ],
        )
        .pip_install("hf-transfer", "grpclib", "requests")
        .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .dockerfile_commands("ENTRYPOINT []")
        .add_local_python_source("cli")
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
            scaledown_window=scaledown_window,
            timeout=timeout,
            region=region,
        )(modal.concurrent(max_inputs=1000)(cls))

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
                *(
                    ["--tokenizer-path", server_config["tokenizer"]]
                    if "tokenizer" in server_config
                    else []
                ),
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


@sglang_cls(gpu="H100!:4")
class SGLang_4xH100(SGLangBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@sglang_cls(gpu="H100!:8", cpu=8)
class SGLang_8xH100(SGLangBase):
    model: str = modal.parameter()
    caller_id: str = modal.parameter(default="")
    server_config: str = modal.parameter(default="{}")


@contextlib.contextmanager
def sglang(
    model: str,
    gpu: str,
    region: str,
    server_config: Optional[Mapping[str, Any]] = None,
    profile: bool = False,
):
    import requests

    if profile:
        raise ValueError("Profiling is not supported for SGLang")

    all_sglang_classes = {
        DEFAULT_DOCKER_TAG: {
            "H100": {
                "us-chicago-1": SGLang,
            },
            "H100:2": {
                "us-chicago-1": SGLang_2xH100,
            },
            "H100:4": {
                "us-chicago-1": SGLang_4xH100,
            },
            "H100:8": {
                "us-chicago-1": SGLang_8xH100,
            },
        },
    }

    docker_tag = server_config.get("docker_tag", DEFAULT_DOCKER_TAG)
    extra_query = {
        "model": model,
        # Sort keys to ensure that this parameter doesn't change between runs
        # with the same SGLang configuration
        "server_config": json.dumps(server_config, sort_keys=True),
        "caller_id": modal.current_function_call_id(),
    }

    # Pick SGLang server class
    try:
        cls = all_sglang_classes[docker_tag][gpu.replace("!", "")][region]
    except KeyError:
        raise ValueError(
            f"Unsupported SGLang configuration: {docker_tag} {gpu} {region}"
        )

    url = cls(model="").start.web_url

    # Wait for SGLang server to start
    print(f"Requesting health check at {url}/health_generate with params {extra_query}")

    num_retries = 3
    for retry in range(num_retries):
        res = requests.get(f"{url}/health_generate", params=extra_query)

        if res.status_code == 200:
            break
        else:
            time.sleep(5)

        if retry == num_retries - 1:
            raise ValueError(
                f"Failed to connect to SGLang instance: {res.status_code} {res.text}"
            )

    print("Connected to SGLang instance")
    yield (url, extra_query, docker_tag)
