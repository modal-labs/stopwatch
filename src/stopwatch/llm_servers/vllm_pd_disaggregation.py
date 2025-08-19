import json
import os
import subprocess
from datetime import datetime, timezone

import modal

from stopwatch.constants import HOURS, VersionDefaults
from stopwatch.resources import (
    hf_cache_volume,
    startup_metrics_dict,
    vllm_cache_volume,
)

from .vllm import PORT, VLLM_PYTHON_BINARY, vllm_image_factory

DECODE_PORT = 8200
PREFILL_PORT = 8100
PROXY_SERVER_SCRIPT = "/root/tools/pd_disaggregation_proxy_server.py"


def vllm_pd_disaggregation_image_factory(
    docker_tag: str = VersionDefaults.VLLM,
) -> modal.Image:
    """
    Create a Modal image for running a vLLM server with PD disaggregation.

    :param: docker_tag: The tag of the vLLM Docker image to use.
    :return: A Modal image for running a vLLM server with PD disaggregation.
    """

    return vllm_image_factory(
        docker_tag,
        extra_dockerfile_commands=[
            f"RUN {VLLM_PYTHON_BINARY} -m pip install quart --ignore-installed",
        ],
    ).add_local_python_source("tools")


class vLLMPDDisaggregationBase:
    """A Modal class that runs a vLLM server."""

    @modal.web_server(port=PORT, startup_timeout=1 * HOURS)
    def start(self) -> None:
        """Start a vLLM server."""

        # Save the startup time to a dictionary so we can measure cold start duration
        startup_metrics_dict[self.caller_id] = datetime.now(timezone.utc).timestamp()

        hf_cache_volume.reload()
        vllm_cache_volume.reload()

        if not self.model:
            msg = "model must be set, e.g. 'meta-llama/Llama-3.1-8B-Instruct'"
            raise ValueError(msg)

        server_config = json.loads(self.server_config)

        # vLLM currently only supports 2-GPU setups for disaggregated prefill. See:
        # https://github.com/vllm-project/vllm/issues/13004
        prefill_devices = "0"
        decode_devices = "1"

        # Start prefill server
        subprocess.Popen(
            " ".join(
                [
                    VLLM_PYTHON_BINARY,
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    self.model,
                    *(
                        ["--tokenizer", server_config["tokenizer"]]
                        if "tokenizer" in server_config
                        else []
                    ),
                    "--kv-transfer-config",
                    '\'{"kv_connector":"SharedStorageConnector","kv_role":"kv_both","kv_connector_extra_config":{"shared_storage_path":"local_storage"}}\'',
                    "--port",
                    str(PREFILL_PORT),
                    *server_config.get("extra_args", []),
                ],
            )
            + f" || {VLLM_PYTHON_BINARY} -m http.server {PREFILL_PORT}",
            env={
                **os.environ,
                **server_config.get("env_vars", {}),
                "CUDA_VISIBLE_DEVICES": prefill_devices,
            },
            shell=True,
        )

        # Start decode server
        subprocess.Popen(
            " ".join(
                [
                    VLLM_PYTHON_BINARY,
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    self.model,
                    *(
                        ["--tokenizer", server_config["tokenizer"]]
                        if "tokenizer" in server_config
                        else []
                    ),
                    "--kv-transfer-config",
                    '\'{"kv_connector":"SharedStorageConnector","kv_role":"kv_both","kv_connector_extra_config":{"shared_storage_path":"local_storage"}}\'',
                    "--port",
                    str(DECODE_PORT),
                    *server_config.get("extra_args", []),
                ],
            )
            + f" || {VLLM_PYTHON_BINARY} -m http.server {DECODE_PORT}",
            env={
                **os.environ,
                **server_config.get("env_vars", {}),
                "CUDA_VISIBLE_DEVICES": decode_devices,
            },
            shell=True,
        )

        # Wait for both servers to start
        import time

        import requests

        for port in [PREFILL_PORT, DECODE_PORT]:
            while True:
                try:
                    requests.get(f"http://localhost:{port}/v1/completions")
                    break
                except requests.exceptions.ConnectionError:
                    time.sleep(1)

        # Run proxy server
        subprocess.Popen(f"{VLLM_PYTHON_BINARY} {PROXY_SERVER_SCRIPT}", shell=True)
