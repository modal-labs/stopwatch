import itertools
import os
import subprocess
import yaml

import click
import modal

from stopwatch.resources import traces_volume
from stopwatch.run_benchmark import all_benchmark_runner_classes


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--model",
    type=str,
    help="Name of the model to benchmark.",
    required=True,
)
@click.option(
    "--data",
    type=str,
    default="prompt_tokens=512,generated_tokens=128",
    help="The data source to use for benchmarking. Depending on the data-type, it should be a path to a data file containing prompts to run (ex: data.txt), a HuggingFace dataset name (ex: 'neuralmagic/LLM_compression_calibration'), or a configuration for emulated data (ex: 'prompt_tokens=128,generated_tokens=128').",
)
@click.option(
    "--data-type",
    type=str,
    default="emulated",
    help="The type of data to use, such as 'emulated', 'file', or 'transformers'. Defaults to 'emulated'.",
)
@click.option(
    "--gpu",
    type=str,
    default="H100",
    help="GPU to run the LLM server on. Defaults to 'H100'.",
)
@click.option(
    "--region",
    type=str,
    default="us-ashburn-1",
    help="Region to run the LLM server on. Defaults to 'us-ashburn-1'.",
)
@click.option(
    "--llm-server-type",
    type=str,
    default="vllm",
    help="LLM server to use (vllm or trtllm).",
)
def run_benchmark(**kwargs):
    cls = all_benchmark_runner_classes[kwargs["region"]]
    fc = cls().run_benchmark.spawn(**kwargs)
    print(f"Benchmark running at {fc.object_id}")


@cli.command()
@click.option(
    "--gpu",
    type=str,
    default="H100!",
    help="GPU to run the LLM server on. Defaults to 'H100!'.",
)
@click.option(
    "--model",
    type=str,
    help="Name of the model to run while profiling LLM.",
    required=True,
)
@click.option(
    "--num-requests",
    type=int,
    help="Number of requests to make to LLM while profiling.",
    default=10,
)
def run_profiler(**kwargs):
    f = modal.Function.from_name("stopwatch", "run_profiler")
    fc = f.spawn(**kwargs)
    print(f"Profiler running at {fc.object_id}...")
    trace_path = fc.get()

    # Download and save trace
    os.makedirs("traces", exist_ok=True)
    local_trace_path = os.path.join("traces", trace_path)

    with open(local_trace_path, "wb") as f:
        for chunk in traces_volume.read_file(trace_path):
            f.write(chunk)

    # Optionally show file in Finder
    answer = input(f"Saved to {trace_path}. Show in Finder? [Y/n] ")

    if answer != "n":
        subprocess.run(["open", "-R", local_trace_path])


@cli.command()
@click.argument("config-path", type=str)
@click.option(
    "--recompute",
    is_flag=True,
    default=False,
    help="Recompute benchmarks that have already been run.",
)
def run_benchmark_suite(config_path: str, recompute: bool = False):
    config = yaml.load(open(config_path), Loader=yaml.SafeLoader)
    benchmarks = []

    for config_spec in config["configs"]:
        keys = []
        values = []

        for key, value in config_spec.items():
            keys.append(key)
            values.append(value if isinstance(value, list) else [value])

        for combination in itertools.product(*values):
            benchmarks.append(dict(zip(keys, combination)))

    f = modal.Function.from_name("stopwatch", "run_benchmark_suite")
    fc = f.spawn(
        benchmarks=benchmarks,
        repeats=config.get("repeats", 1),
        recompute=recompute,
    )

    print("Running benchmarks (you may safely CTRL+C)...")
    fc.get()

    # Optionally open the datasette UI
    answer = input("All benchmarks have finished. Open the datasette UI? [Y/n] ")

    if answer != "n":
        url = modal.Cls.from_name("stopwatch", "DatasetteRunner")().start.web_url
        subprocess.run(["open", url])


if __name__ == "__main__":
    cli()
