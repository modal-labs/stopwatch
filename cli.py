from typing import Optional
import itertools
import json
import os
import subprocess
import yaml

import click
import modal

from stopwatch.resources import app, traces_volume
from stopwatch.run_benchmark import all_benchmark_runner_classes


@click.group()
def cli():
    pass


@app.local_entrypoint()
def run_benchmark(
    model: str,
    data: str = "prompt_tokens=512,generated_tokens=128",
    gpu: str = "H100",
    server_region: str = "us-chicago-1",
    client_region: str = "us-chicago-1",
    llm_server_type: str = "vllm",
    llm_server_config: Optional[str] = None,
    rate_type: str = "synchronous",
    rate: Optional[float] = None,
):
    cls = all_benchmark_runner_classes[client_region]

    if llm_server_config is not None:
        try:
            llm_server_config = json.loads(llm_server_config)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON for --llm-server-config")

    if rate_type == "constant" and rate is None:
        raise ValueError("--rate is required when --rate-type is 'constant'")

    results = cls().run_benchmark.remote(
        llm_server_type=llm_server_type,
        model=model,
        rate_type=rate_type,
        data=data,
        gpu=gpu,
        server_region=server_region,
        llm_server_config=llm_server_config,
        rate=rate,
    )

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def run_profiler(model: str, gpu: str = "H100", num_requests: int = 10):
    f = modal.Function.from_name("stopwatch", "run_profiler")
    fc = f.spawn(model=model, gpu=gpu, num_requests=num_requests)
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
    "--dry-run",
    is_flag=True,
    default=False,
    help="Run the benchmark suite without actually running any benchmarks.",
)
@click.option(
    "--ignore-previous-errors",
    is_flag=True,
    default=False,
    help="Ignore errors when checking the results of previous function calls.",
)
@click.option(
    "--recompute",
    is_flag=True,
    default=False,
    help="Recompute benchmarks that have already been run.",
)
def run_benchmark_suite(
    config_path: str,
    dry_run: bool = False,
    ignore_previous_errors: bool = False,
    recompute: bool = False,
):
    config = yaml.load(open(config_path), Loader=yaml.SafeLoader)
    benchmarks = []

    for config_spec in config["configs"]:
        config_spec = {**config.get("base_config", {}), **config_spec}
        keys = []
        values = []

        for key, value in config_spec.items():
            keys.append(key)
            values.append(value if isinstance(value, list) else [value])

        for combination in itertools.product(*values):
            benchmark_config = dict(zip(keys, combination))

            if "region" in benchmark_config:
                benchmark_config["server_region"] = benchmark_config["region"]
                benchmark_config["client_region"] = benchmark_config["region"]
                del benchmark_config["region"]

            benchmarks.append(benchmark_config)

    if "id" not in config:
        raise ValueError("'id' is required in the config")

    f = modal.Function.from_name("stopwatch", "run_benchmark_suite")
    fc = f.spawn(
        benchmarks=benchmarks,
        id=config["id"],
        repeats=config.get("repeats", 1),
        dry_run=dry_run,
        ignore_previous_errors=ignore_previous_errors,
        recompute=recompute,
    )

    print("Running benchmarks (you may safely CTRL+C)...")
    fc.get()

    # Optionally open the datasette UI
    answer = input("All benchmarks have finished. Open the datasette UI? [Y/n] ")

    if answer != "n":
        url = modal.Cls.from_name("stopwatch", "DatasetteRunner")().start.web_url
        url += f"/stopwatch/-/query?sql=select+*+from+{config['id'].replace('-', '_')}"
        subprocess.run(["open", url])


if __name__ == "__main__":
    cli()
