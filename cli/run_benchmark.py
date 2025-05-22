from typing import Optional
import json
import sys

from stopwatch.db import RateType
from stopwatch.llm_server import deploy_llm_server_cls
from stopwatch.resources import app
from stopwatch.run_benchmark import (
    deploy_benchmark_runner_cls,
    get_benchmark_runner_class_name,
)

CLIENT_REGION = "us-chicago-1"
GPU = "H100"
LLM_SERVER_TYPE = "vllm"
LLM_SERVER_CONFIG = {}
SERVER_REGION = "us-chicago-1"


def local_entrypoint_with_dynamic_classes():
    def decorator(fn):
        # Parse command line arguments
        args = {
            k: globals()[k.replace("-", "_").upper()]
            for k in [
                "client-region",
                "gpu",
                "llm-server-type",
                "llm-server-config",
                "server-region",
            ]
        }

        for key in args:
            if f"--{key}" in sys.argv and sys.argv.index(f"--{key}") + 1 < len(
                sys.argv
            ):
                args[key] = sys.argv[sys.argv.index(f"--{key}") + 1]

        # Create client class
        deploy_benchmark_runner_cls(args["client-region"])

        # Deploy server class
        deploy_llm_server_cls(
            args["llm-server-type"],
            args["gpu"],
            args["server-region"],
            args["llm-server-config"],
        )

        return app.local_entrypoint()(fn)

    return decorator


@local_entrypoint_with_dynamic_classes()
def run_benchmark(
    model: str,
    data: str = "prompt_tokens=512,output_tokens=128",
    gpu: str = GPU,
    server_region: str = CLIENT_REGION,
    client_region: str = SERVER_REGION,
    llm_server_type: str = LLM_SERVER_TYPE,
    duration: Optional[float] = 120,
    llm_server_config: Optional[str] = LLM_SERVER_CONFIG,
    client_config: Optional[str] = None,
    rate_type: str = RateType.SYNCHRONOUS.value,
    rate: Optional[float] = None,
):
    cls_name = get_benchmark_runner_class_name(client_region)

    if cls_name not in app.registered_classes:
        raise ValueError(f"Benchmark runner class {cls_name} not found")

    cls = app.registered_classes[cls_name]

    if llm_server_config is not None:
        try:
            llm_server_config = json.loads(llm_server_config)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON for --llm-server-config")

    if client_config is not None:
        try:
            client_config = json.loads(client_config)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON for --client-config")

    if rate_type == RateType.CONSTANT.value and rate is None:
        raise ValueError(
            f"--rate is required when --rate-type is {RateType.CONSTANT.value}"
        )

    results = cls().run_benchmark.remote(
        llm_server_type=llm_server_type,
        model=model,
        rate_type=rate_type,
        data=data,
        gpu=gpu,
        server_region=server_region,
        duration=duration,
        llm_server_config=llm_server_config,
        client_config=client_config,
        rate=rate,
    )

    return json.dumps(results, indent=2)
