import json
import uuid
from pathlib import Path

import modal

from stopwatch.benchmark.dynamic import create_dynamic_benchmark_runner_cls
from stopwatch.db import RateType
from stopwatch.llm_servers import llm_server
from stopwatch.resources import app
from stopwatch.llm_servers.dynamic import create_dynamic_llm_server_cls


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

    if llm_server_config is not None:
        try:
            llm_server_config = json.loads(llm_server_config)
        except json.JSONDecodeError as e:
            msg = "Invalid JSON for --llm-server-config"
            raise ValueError(msg) from e
    else:
        llm_server_config = {}

    if client_config is not None:
        try:
            client_config = json.loads(client_config)
        except json.JSONDecodeError as e:
            msg = "Invalid JSON for --client-config"
            raise ValueError(msg) from e
    else:
        client_config = {}

    if rate_type == RateType.CONSTANT.value and rate is None:
        msg = f"--rate is required when --rate-type is {RateType.CONSTANT.value}"
        raise ValueError(msg)

    name = uuid.uuid4().hex[:4]
    client_cls = create_dynamic_benchmark_runner_cls(name, client_region)
    server_cls = create_dynamic_llm_server_cls(
        name,
        model,
        gpu=gpu,
        llm_server_type=llm_server_type,
        region=server_region,
        llm_server_config=llm_server_config,
    )

    with modal.enable_output(), app.run(detach=detach):
        with llm_server(
            server_cls().start.get_web_url(),
            llm_server_type=llm_server_type,
            model=model,
            gpu=gpu,
            region=server_region,
            server_config=llm_server_config,
        ) as (llm_server_url,):
            # TODO(jack): Set remove_from_body for tokasaurus
            # client_config["extra_query"] = extra_query

            # TODO(jack): Start the benchmark runner before the LLM server is ready
            results = client_cls().run_benchmark.remote(
                endpoint=f"{llm_server_url}/v1",
                model=model,
                rate_type=rate_type,
                data=data,
                duration=duration,
                client_config=client_config,
                rate=rate,
            )

        with Path(output_path).open("w") as f:
            json.dump(results, f, indent=2)
