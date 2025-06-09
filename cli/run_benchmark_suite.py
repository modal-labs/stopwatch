import itertools
import os
import yaml

import modal

from stopwatch.resources import app
from stopwatch.run_benchmark_suite import run_benchmark_suite


def load_benchmarks_from_config(config_path: str):
    config = yaml.load(open(config_path), Loader=yaml.SafeLoader)

    if "id" not in config:
        raise ValueError("'id' is required in the config")

    benchmarks = []

    for config_spec in config.get("configs", [] if "files" in config else [{}]):
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


@app.local_entrypoint()
def run_benchmark_suite_cli(config_path: str, fast_mode: bool = False):
    benchmarks, id, version, repeats = load_benchmarks_from_config(config_path)

    run_benchmark_suite.remote(
        benchmarks=benchmarks,
        id=id,
        version=version,
        repeats=repeats,
        fast_mode=fast_mode,
    )

    print()
    print("To view the results of this benchmark suite, you may:")

    # Provide link to Datasette UI
    try:
        datasette_url = modal.Cls.from_name(
            "stopwatch", "DatasetteRunner"
        )().start.get_web_url()

        print("- Open the Datasette UI at:")
        print(
            f"   {datasette_url}/stopwatch/-/query?sql=select+*+from+{id.replace('-', '_')}_averaged+where+rate_type+%21%3D+\"throughput\""
        )
    except Exception:
        print("- Deploy the Datasette UI with:")
        print("   modal deploy -m stopwatch")

    # Provide path to JSONL file
    print("- Download the results JSONL file with:")
    print(f"   modal volume get stopwatch-results {id}.jsonl")
