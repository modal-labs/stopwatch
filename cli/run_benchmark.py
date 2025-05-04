from typing import Optional
import json

from stopwatch.db import RateType
from stopwatch.resources import app
from stopwatch.run_benchmark import all_benchmark_runner_classes


@app.local_entrypoint()
def run_benchmark(
    model: str,
    data: str = "prompt_tokens=512,output_tokens=128",
    gpu: str = "H100",
    server_region: str = "us-chicago-1",
    client_region: str = "us-chicago-1",
    llm_server_type: str = "vllm",
    duration: Optional[float] = 120,
    llm_server_config: Optional[str] = None,
    client_config: Optional[str] = None,
    rate_type: str = RateType.SYNCHRONOUS.value,
    rate: Optional[float] = None,
):
    cls = all_benchmark_runner_classes[client_region]

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
