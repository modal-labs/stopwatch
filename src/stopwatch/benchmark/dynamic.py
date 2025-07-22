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


def BenchmarkRunnerClassFactory(name: str) -> type:  # noqa: N802
    """Create a benchmark runner class with a given name."""

    return type(
        name,
        (GuideLLMRunner,),
        {},
    )


def create_dynamic_benchmark_runner_cls(name: str, region: str | None = None) -> type:
    """Create a dynamic benchmark runner class that can be run on Modal."""

    if region is None:
        benchmark_runner_name = f"{name}"
    else:
        benchmark_runner_name = f"{name}_{region.replace('-','_')}"

    if not benchmark_runner_name.startswith("BenchmarkRunner_"):
        benchmark_runner_name = f"BenchmarkRunner_{benchmark_runner_name}"

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
    """
    When Stopwatch is run, classes will be created dynamically in order to meet the
    needs of the benchmark being run. Modal will then need to call these classes once
    the benchmark is run. This function allows us to dynamically create these classes
    once Stopwatch has already been deployed.
    """

    if name in globals():
        return globals()[name]

    if name.startswith("BenchmarkRunner_"):
        return BenchmarkRunnerClassFactory(name)

    msg = f"No attribute {name}"
    raise AttributeError(msg)
