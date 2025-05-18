import itertools
import os
import subprocess
import yaml

import click
import modal

from stopwatch.resources import results_volume


def load_benchmarks_from_config(config_path: str):
    config = yaml.load(open(config_path), Loader=yaml.SafeLoader)

    if "id" not in config:
        raise ValueError("'id' is required in the config")

    benchmarks = []

    for config_spec in config.get("configs", []):
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

    for file in config.get("files", []):
        file_benchmarks, _, _, _ = load_benchmarks_from_config(
            os.path.join(os.path.dirname(config_path), file)
        )
        benchmarks.extend(file_benchmarks)

    return benchmarks, config["id"], config.get("version", 1), config.get("repeats", 1)


@click.command()
@click.argument("config-path", type=str)
@click.option(
    "--disable-safe-mode",
    is_flag=True,
    help="Disable safe mode, which runs all of your benchmarks once without repeats before continuing. This slows down your benchmark runs, but ensures you don't waste money on invalid configurations.",
)
@click.option(
    "--ignore-previous-errors",
    is_flag=True,
    help="Ignore errors when checking the results of previous function calls.",
)
@click.option(
    "--recompute",
    is_flag=True,
    help="Recompute benchmarks that have already been run.",
)
def run_benchmark_suite(
    config_path: str,
    disable_safe_mode: bool = False,
    ignore_previous_errors: bool = False,
    recompute: bool = False,
):
    benchmarks, id, version, repeats = load_benchmarks_from_config(config_path)

    f = modal.Function.from_name("stopwatch", "run_benchmark_suite")
    fc = f.spawn(
        benchmarks=benchmarks,
        id=id,
        version=version,
        repeats=repeats,
        disable_safe_mode=disable_safe_mode,
        ignore_previous_errors=ignore_previous_errors,
        recompute=recompute,
    )

    print("Running benchmarks (you may safely CTRL+C)...")
    fc.get()

    # Optionally open the datasette UI
    answer = input("All benchmarks have finished. Open the datasette UI? [Y/n] ")

    if answer != "n":
        url = modal.Cls.from_name("stopwatch", "DatasetteRunner")().start.web_url
        url += f"/stopwatch/-/query?sql=select+*+from+{id.replace('-', '_')}_averaged"
        subprocess.run(["open", url])

    # Optionally save JSONL file
    answer = input("Download results JSONL file and show in Finder? [Y/n] ")

    if answer != "n":
        os.makedirs("results", exist_ok=True)
        local_results_path = os.path.join("results", f"{id}.jsonl")

        with open(local_results_path, "wb") as f:
            for chunk in results_volume.read_file(f"{id}.jsonl"):
                f.write(chunk)

        subprocess.run(["open", "-R", local_results_path])


if __name__ == "__main__":
    run_benchmark_suite()
