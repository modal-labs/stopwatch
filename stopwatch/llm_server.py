from typing import Any, Mapping, Optional
import contextlib
import json
import time
import uuid

import modal

from .constants import VersionDefaults
from .sglang_runner import sglang_classes
from .tensorrt_llm_runner import tensorrt_llm_classes
from .vllm_runner import vllm_classes


SGLANG = "sglang"
TENSORRT_LLM = "tensorrt-llm"
VLLM = "vllm"


@contextlib.contextmanager
def llm_server(
    llm_server_type: str,
    *,
    model: str,
    gpu: str,
    region: str,
    server_config: Optional[Mapping[str, Any]] = None,
    profile: bool = False,
):
    import requests

    if profile and llm_server_type != VLLM:
        raise ValueError("Profiling is only supported for vLLM")

    llm_server_classes = {
        SGLANG: sglang_classes,
        TENSORRT_LLM: tensorrt_llm_classes,
        VLLM: vllm_classes,
    }

    llm_server_version = server_config.get(
        "version", VersionDefaults.LLM_SERVERS[llm_server_type]
    )

    extra_query = {
        "model": model,
        # Sort keys to ensure that this parameter doesn't change between runs
        # with the same vLLM configuration
        "server_config": json.dumps(server_config, sort_keys=True),
        "caller_id": modal.current_function_call_id(),
    }

    # Pick LLM server class
    try:
        cls = llm_server_classes[llm_server_type][llm_server_version][
            gpu.replace("!", "")
        ][region]
    except KeyError:
        raise ValueError(
            f"Unsupported configuration: {llm_server_type} {llm_server_version} {gpu} {region}"
        )

    url = cls(model="").start.web_url

    if llm_server_type == SGLANG:
        health_url = f"{url}/health_generate"
    elif llm_server_type == TENSORRT_LLM:
        health_url = f"{url}/health"
    elif llm_server_type == VLLM:
        health_url = f"{url}/metrics"

    # Wait for LLM server to start
    print(f"Requesting health check at {health_url} with params {extra_query}")

    num_retries = 3
    for retry in range(num_retries):
        try:
            res = requests.get(health_url, params=extra_query)
        except requests.HTTPError as e:
            print(f"Error requesting metrics: {e}")
            extra_query["caller_id"] = str(uuid.uuid4())
            continue

        if res.status_code == 404:
            # If this endpoint returns a 404, it is because the LLM server startup
            # command returned a nonzero exit code, resulting in this request being
            # routed to the fallback Python http.server module. This means that
            # the LLM server crashed on startup.

            raise Exception(
                f"The {llm_server_type} server has crashed, likely due to a "
                "misconfiguration. Please investigate this crash before proceeding."
            )

        if (
            llm_server_type == SGLANG or llm_server_type == TENSORRT_LLM
        ) and res.status_code == 200:
            break
        elif llm_server_type == VLLM and "vllm:gpu_cache_usage_perc" in res.text:
            break

        if retry == num_retries - 1:
            raise ValueError(
                f"Failed to connect to LLM server after {num_retries} retries: {res.status_code} {res.text}"
            )

        time.sleep(5)

    print("Connected to LLM server")

    if profile:
        requests.post(f"{url}/start_profile", params=extra_query)

    try:
        yield (url, extra_query)
    finally:
        if profile:
            requests.post(f"{url}/stop_profile", params=extra_query)
