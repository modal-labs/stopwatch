import itertools
from pathlib import Path
from typing import Any

import modal
import yaml

from stopwatch.resources import app
from stopwatch.run_benchmark_suite import run_benchmark_suite


def load_benchmarks_from_config(
    config_path: Path,
) -> tuple[list[dict[str, Any]], str, int, int]:
    """
    Load benchmark configurations from a YAML config file.

    :param: config_path: The path to the YAML file containing the benchmark
        configurations.
    :return: A tuple containing all specified benchmark configurations, the suite ID,
        the suite version, and the number of times the suite should be repeated.
    """

    config = yaml.load(config_path.open(), Loader=yaml.SafeLoader)

    if "id" not in config:
        msg = "'id' is required in the config"
        raise ValueError(msg)

    benchmarks = []

    for config_spec in config.get("configs", [] if "files" in config else [{}]):
        full_config_spec = {**config.get("base_config", {}), **config_spec}
        keys = []
        values = []

        for key, value in full_config_spec.items():
            keys.append(key)
            values.append(value if isinstance(value, list) else [value])

        for combination in itertools.product(*values):
            benchmark_config = dict(zip(keys, combination, strict=False))

            if "region" in benchmark_config:
                benchmark_config["server_region"] = benchmark_config["region"]
                benchmark_config["client_region"] = benchmark_config["region"]
                del benchmark_config["region"]

            benchmarks.append(benchmark_config)

    for file in config.get("files", []):
        file_benchmarks, _, _, _ = load_benchmarks_from_config(
            config_path.parent / file,
        )
        benchmarks.extend(file_benchmarks)

    return benchmarks, config["id"], config.get("version", 1), config.get("repeats", 1)


@app.local_entrypoint()
def run_benchmark_suite_cli(config_path: str, *, fast_mode: bool = False) -> None:
    """
    Run a benchmark suite.

    :param: config_path: The path to the YAML file containing the benchmark
        configurations.
    :param: fast_mode: Whether to run the benchmark suite in fast mode, which will run
        more benchmarks in parallel, incurring extra costs since LLM servers are not
        reused between benchmarks. Disabled by default.
    """

    benchmarks, suite_id, version, repeats = load_benchmarks_from_config(
        Path(config_path),
    )

    run_benchmark_suite.remote(
        benchmarks=benchmarks,
        suite_id=suite_id,
        version=version,
        repeats=repeats,
        fast_mode=fast_mode,
    )

    print()
    print("To view the results of this benchmark suite, you may:")

    # Provide link to Datasette UI
    try:
        datasette_url = modal.Cls.from_name(
            "stopwatch",
            "DatasetteRunner",
        )().start.get_web_url()

        print("- Open the Datasette UI at:")
        print(
            f"   {datasette_url}/stopwatch/-/query?sql=select+*+from+"
            f"{suite_id.replace('-', '_')}_averaged+where+rate_type+%21%3D+"
            '"throughput"',
        )
    except Exception:  # noqa: BLE001
        print("- Deploy the Datasette UI with:")
        print("   modal deploy -m stopwatch")

    # Provide path to JSONL file
    print("- Download the results JSONL file with:")
    print(f"   modal volume get stopwatch-results {suite_id}.jsonl")
