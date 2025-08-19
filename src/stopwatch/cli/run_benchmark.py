import json
import uuid
from pathlib import Path
from typing import Annotated

import modal
import typer

from stopwatch.benchmark import create_dynamic_benchmark_runner_cls
from stopwatch.constants import LLMServerType, RateType
from stopwatch.llm_servers import create_dynamic_llm_server_cls
from stopwatch.resources import app


def config_callback(config: str | None) -> dict:
    """Parse JSON config strings into dicts."""

    if isinstance(config, dict):
        return config

    if config is None:
        return {}

    try:
        return json.loads(config)
    except json.JSONDecodeError as err:
        msg = "Must be a valid JSON string"
        raise typer.BadParameter(msg) from err


def rate_type_callback(ctx: typer.Context, rate_type: RateType) -> RateType:
    """Require rate to be provided when rate_type is constant."""

    if rate_type == RateType.constant and ctx.params.get("rate") is None:
        msg = "Rate must be provided when rate_type is constant"
        raise typer.BadParameter(msg)

    return rate_type


def run_benchmark_cli(
    model: str,
    llm_server_type: LLMServerType,
    *,
    output_path: str | None = "results.json",
    detach: bool = False,
    data: str = "prompt_tokens=512,output_tokens=128",
    gpu: str = "H100",
    server_region: str | None = None,
    client_region: str | None = None,
    server_cloud: str | None = None,
    duration: float | None = 120,
    llm_server_config: Annotated[
        str | None,
        typer.Option(callback=config_callback),
    ] = None,
    client_config: Annotated[str | None, typer.Option(callback=config_callback)] = None,
    rate_type: Annotated[
        RateType,
        typer.Option(callback=rate_type_callback),
    ] = RateType.synchronous,
    rate: float | None = None,
) -> list[dict]:
    """Run a benchmark."""

    name = uuid.uuid4().hex[:4]
    client_cls = create_dynamic_benchmark_runner_cls(name, client_region)
    server_cls = create_dynamic_llm_server_cls(
        name,
        model,
        gpu=gpu,
        llm_server_type=llm_server_type,
        cloud=server_cloud,
        region=server_region,
        llm_server_config=llm_server_config,
    )

    with modal.enable_output(), app.run(detach=detach):
        results = client_cls().run_benchmark.remote(
            endpoint=f"{server_cls().start.get_web_url()}/v1",
            model=model,
            rate_type=rate_type,
            data=data,
            duration=duration,
            client_config=client_config,
            rate=rate,
        )

        if output_path is not None:
            with Path(output_path).open("w") as f:
                json.dump(results, f, indent=2)

    return results
