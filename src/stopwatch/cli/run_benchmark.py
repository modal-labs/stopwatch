import json
from pathlib import Path

import modal

from stopwatch.db import RateType
from stopwatch.resources import app
from stopwatch.run_benchmark import all_benchmark_runner_classes


def run_benchmark_cli(
    model: str,
    *,
    output_path: str = "results.json",
    detach: bool = False,
    data: str = "prompt_tokens=512,output_tokens=128",
    gpu: str = "H100",
    server_region: str = "us-chicago-1",
    client_region: str = "us-chicago-1",
    llm_server_type: str = "vllm",
    duration: float | None = 120,
    llm_server_config: str | None = None,
    client_config: str | None = None,
    rate_type: str = RateType.SYNCHRONOUS.value,
    rate: float | None = None,
) -> None:
    """Run a benchmark."""

    with modal.enable_output(), app.run(detach=detach):
        cls = all_benchmark_runner_classes[client_region]

        if llm_server_config is not None:
            try:
                llm_server_config = json.loads(llm_server_config)
            except json.JSONDecodeError as e:
                msg = "Invalid JSON for --llm-server-config"
                raise ValueError(msg) from e

        if client_config is not None:
            try:
                client_config = json.loads(client_config)
            except json.JSONDecodeError as e:
                msg = "Invalid JSON for --client-config"
                raise ValueError(msg) from e

        if rate_type == RateType.CONSTANT.value and rate is None:
            msg = f"--rate is required when --rate-type is {RateType.CONSTANT.value}"
            raise ValueError(msg)

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

        with Path(output_path).open("w") as f:
            json.dump(results, f, indent=2)
