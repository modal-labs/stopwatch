import contextlib
import json
import logging
import time
import uuid
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from typing import Any

import modal

from stopwatch.constants import VersionDefaults
from stopwatch.resources import startup_metrics_dict

from .constants import SGLANG, TENSORRT_LLM, TOKASAURUS, VLLM, VLLM_PD_DISAGGREGATION
from .dynamic import create_dynamic_llm_server_cls

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@contextlib.contextmanager
def llm_server(
    url: str,
    *,
    llm_server_type: str,
    model: str,
    gpu: str,
    region: str,
    server_config: Mapping[str, Any] | None = None,
    profile: bool = False,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """
    Context manager that starts an LLM server and yields its URL and extra query
    parameters.

    :param: llm_server_type: The type of LLM server to start, either 'sglang',
        'tensorrt-llm', 'tokasaurus', or 'vllm'.
    :param: model: The model to start the server with.
    :param: gpu: The GPU to start the server with.
    :param: region: The region to start the server in.
    :param: server_config: Extra configuration to start the server with.
    :param: profile: Whether to profile the server.

    :yield: A tuple containing the URL of the server and extra query parameters that
        need to be included in requests to the server.
    """
    import requests

    if profile and llm_server_type != VLLM:
        msg = "Profiling is only supported for vLLM"
        raise ValueError(msg)

    llm_health_routes = {
        SGLANG: "/health_generate",
        TENSORRT_LLM: "/health",
        TOKASAURUS: "/ping",
        VLLM: "/metrics",
        VLLM_PD_DISAGGREGATION: "/ping",
    }

    health_url = f"{url}{llm_health_routes[llm_server_type]}"
    queue_time = datetime.now(timezone.utc).timestamp()

    # Wait for LLM server to start
    logger.info(
        "Requesting health check at %s",
        health_url,
    )

    num_retries = 3
    for retry in range(num_retries):
        # Ensure we don't hit the default limit of 30 redirects when waiting on
        # functions with a long startup time.
        session = requests.Session()
        session.max_redirects = 9999

        try:
            res = session.get(health_url)
        except requests.HTTPError:
            logger.exception("Error requesting metrics")
            continue

        if res.status_code == 404:  # noqa: PLR2004
            # If this endpoint returns a 404, it is because the LLM server startup
            # command returned a nonzero exit code, resulting in this request being
            # routed to the fallback Python http.server module. This means that
            # the LLM server crashed on startup.

            msg = (
                f"The {llm_server_type} server has crashed, likely due to a "
                "misconfiguration. Please investigate this crash before proceeding.",
            )
            raise Exception(msg)

        if (
            llm_server_type
            in (SGLANG, TENSORRT_LLM, TOKASAURUS, VLLM_PD_DISAGGREGATION)
            and res.status_code == 200  # noqa: PLR2004
        ) or (llm_server_type == VLLM and "vllm:gpu_cache_usage_perc" in res.text):
            break

        if retry == num_retries - 1:
            msg = (
                f"Failed to connect to LLM server after {num_retries} retries: "
                f"{res.status_code} {res.text}",
            )
            raise ValueError(msg)

        time.sleep(5)

    logger.info("Connected to LLM server")
    connection_time = datetime.now(timezone.utc).timestamp()
    queue_duration = connection_time - queue_time

    # TODO(jack): Fix
    # if (
    #     container_start_time := startup_metrics_dict.get(extra_query["caller_id"])
    # ) is not None:
    #     cold_start_duration = connection_time - container_start_time
    #     queue_duration -= cold_start_duration
    # else:
    #     cold_start_duration = None

    if profile:
        requests.post(f"{url}/start_profile")

    try:
        yield (
            url,
            # TODO(jack): Fix
            # {
            #     "queue_duration": queue_duration,
            #     "cold_start_duration": cold_start_duration,
            # },
        )
    finally:
        if profile:
            requests.post(f"{url}/stop_profile")
