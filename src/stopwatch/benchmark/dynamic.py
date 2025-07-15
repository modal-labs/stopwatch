from stopwatch.resources import app, hf_secret, results_volume

from .guidellm import (
    NUM_CORES,
    RESULTS_PATH,
    SCALEDOWN_WINDOW,
    TIMEOUT,
    GuideLLMRunner,
    guidellm_image,
)

MEMORY = 1 * 1024


def BenchmarkRunnerClassFactory(name: str):  # noqa: N802
    return type(
        name,
        (GuideLLMRunner,),
        {},
    )


def create_dynamic_benchmark_runner_cls(name: str, region: str | None = None):
    if region is None:
        benchmark_runner_name = f"BenchmarkRunner_{name}"
    else:
        benchmark_runner_name = f"BenchmarkRunner_{name}_{region.replace('-','_')}"

    return app.cls(
        image=guidellm_image,
        secrets=[hf_secret],
        volumes={RESULTS_PATH: results_volume},
        cpu=NUM_CORES,
        memory=MEMORY,
        scaledown_window=SCALEDOWN_WINDOW,
        timeout=TIMEOUT,
        region=region,
    )(
        BenchmarkRunnerClassFactory(benchmark_runner_name),
    )


def __getattr__(name: str):  # noqa: ANN202
    if name in globals():
        return globals()[name]

    if name.startswith("BenchmarkRunner_"):
        return BenchmarkRunnerClassFactory(name)

    msg = f"No attribute {name}"
    raise AttributeError(msg)
