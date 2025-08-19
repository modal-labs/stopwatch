import json
from pathlib import Path

import modal

from stopwatch.benchmark import GuideLLM
from stopwatch.constants import RateType
from stopwatch.resources import app


def benchmark_cli(
    *,
    endpoint: str,
    model: str,
    output_path: str | None = "results.json",
    detach: bool = False,
    rate_type: RateType = RateType.synchronous,
    data: str = "prompt_tokens=128,output_tokens=128",
    duration: float | None = 120,
    client_config: str | None = None,
    rate: float | None = None,
    region: str | None = None,
) -> list[dict]:
    """Benchmark an OpenAI-compatible LLM server using GuideLLM."""

    with modal.enable_output(), app.run(detach=detach):
        print(f"Running benchmark on {endpoint}...")

        results = GuideLLM.with_options(region=region)().run_benchmark.remote(
            endpoint,
            model,
            rate_type,
            data,
            duration,
            client_config,
            rate,
        )

        if output_path is not None:
            with Path(output_path).open("w") as f:
                json.dump(results, f, indent=2)

        print(f"Results saved to {output_path}")

    return results
