from typing import Any, Mapping, Optional
import base64
import contextlib
import json
import time
import uuid

import modal

from .constants import VersionDefaults, SECONDS, MINUTES, HOURS
from .resources import app, hf_cache_volume, hf_secret, traces_volume
from .sglang_runner import SGLangBase, sglang_image_factory
from .tensorrt_llm_runner import TensorRTLLMBase, tensorrt_llm_image_factory
from .vllm_runner import vLLMBase, vllm_image_factory


SGLANG = "sglang"
TENSORRT_LLM = "tensorrt-llm"
VLLM = "vllm"

HF_CACHE_PATH = "/cache"
TRACES_PATH = "/traces"


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
    server_cls_name = get_llm_server_class_name(
        llm_server_type, gpu, region, server_config
    )

    print(app.registered_classes)
    print(app.registered_functions)
    print(app.registered_web_endpoints)

    if server_cls_name not in app.registered_classes:
        raise ValueError(
            f"Unsupported configuration: {llm_server_type} {llm_server_version} {gpu} {region}"
        )

    server_cls = app.registered_classes[server_cls_name]
    url = server_cls(model="").start.get_web_url()

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
        # Ensure we don't hit the default limit of 30 redirects when waiting on
        # functions with a long startup time.
        session = requests.Session()
        session.max_redirects = 9999

        try:
            res = session.get(health_url, params=extra_query)
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


def get_cpu_count(gpu: str) -> int:
    return {
        "H100:8": 32,
        "L40S": 8,
        "L40S:4": 8,
    }.get(gpu, 4)


def get_llm_server_class_name(
    llm_server_type: str,
    gpu: str,
    region: str,
    server_config: Mapping[str, Any],
):
    server_type = (
        "vLLM"
        if llm_server_type == "vllm"
        else (
            "SGLang"
            if llm_server_type == "sglang"
            else (
                "TensorRT_LLM" if llm_server_type == "tensorrt-llm" else llm_server_type
            )
        )
    )

    server_gpu = (f"{gpu.split(':')[1]}x" if ":" in gpu else "") + gpu.split(":")[
        0
    ].replace("!", "")

    server_region = region.upper().replace("-", "_")
    server_cls_name = f"{server_type}_{server_gpu}_{server_region}"

    if "cpu" in server_config:
        server_cls_name += f"_{server_config['cpu']}cores"

    if "memory" in server_config:
        server_cls_name += f"_{server_config['memory']}MB"

    if "version" in server_config:
        server_cls_name += (
            f"_{base64.b32encode(server_config['version'].encode()).decode()}"
        )

    return server_cls_name


def LLMServerClassFactory(name: str, llm_server_type: str):
    base_cls = {SGLANG: SGLangBase, TENSORRT_LLM: TensorRTLLMBase, VLLM: vLLMBase}[
        llm_server_type.lower()
    ]

    model = modal.parameter()
    caller_id = modal.parameter(default="")
    server_config = modal.parameter(default="{}")

    return type(
        name,
        (base_cls,),
        {
            "model": model,
            "caller_id": caller_id,
            "server_config": server_config,
            "__annotations__": {"model": str, "caller_id": str, "server_config": str},
        },
    )


def deploy_llm_server_cls(
    server_type: str,
    gpu: str,
    region: str,
    server_config: Mapping[str, Any],
):
    image_factory_fn = {
        SGLANG: sglang_image_factory,
        TENSORRT_LLM: tensorrt_llm_image_factory,
        VLLM: vllm_image_factory,
    }.get(server_type)

    if image_factory_fn is None:
        raise ValueError(f"Unsupported LLM server type: {server_type}")

    default_cpu_count = {
        "H100:8": 32,
        "L40S": 8,
        "L40S:4": 8,
    }.get(gpu, 4)

    default_memory = {
        "H100:8": 64 * 1024,
        "L40S": 8 * 1024,
        "L40S:4": 8 * 1024,
    }.get(gpu, 4 * 1024)

    default_scaledown_window = {
        "vLLM": 30 * SECONDS,
        "SGLang": 2 * MINUTES,
        "TensorRT-LLM": 30 * SECONDS,
    }.get(server_type, 30 * SECONDS)

    server_cls_name = get_llm_server_class_name(server_type, gpu, region, server_config)

    if server_cls_name in app.registered_classes:
        return

    # Deploy the newly created class
    app.cls(
        image=image_factory_fn(server_config.get("version")),
        secrets=[hf_secret],
        gpu=gpu.replace("H100", "H100!"),
        volumes={HF_CACHE_PATH: hf_cache_volume, TRACES_PATH: traces_volume},
        cpu=server_config.get("cpu", default_cpu_count),
        memory=server_config.get("memory", default_memory),
        scaledown_window=server_config.get(
            "scaledown_window", default_scaledown_window
        ),
        timeout=1 * HOURS,
        region=region,
    )(
        modal.concurrent(max_inputs=1000)(
            LLMServerClassFactory(server_cls_name, server_type)
        )
    )


def __getattr__(name: str):
    names = {
        SGLANG: "SGLang",
        TENSORRT_LLM: "TensorRT_LLM",
        VLLM: "vLLM",
    }

    for server_type, prefix in names.items():
        if name.lower().startswith(prefix.lower()):
            return LLMServerClassFactory(name, server_type)

    raise AttributeError(f"No attribute {name}")
