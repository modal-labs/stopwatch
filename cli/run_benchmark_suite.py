import itertools
import os
import sys
import yaml

import modal

from stopwatch.llm_server import deploy_llm_server_cls
from stopwatch.resources import app
from stopwatch.run_benchmark import deploy_benchmark_runner_cls
from stopwatch.run_benchmark_suite import run_benchmark_suite


HF_CACHE_PATH = "/cache"
TRACES_PATH = "/traces"


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


def local_entrypoint_with_dynamic_classes():
    def decorator(fn):
        if "--config-path" not in sys.argv:
            raise ValueError("config-path is required")
        elif sys.argv.index("--config-path") == len(sys.argv) - 1:
            raise ValueError("config-path must be followed by a path")

        config_path = sys.argv[sys.argv.index("--config-path") + 1]
        benchmarks, _, _, _ = load_benchmarks_from_config(config_path)

        # Deploy a class for each unique client and server configuration
        for benchmark in benchmarks:
            gpu = benchmark["gpu"]
            region = benchmark["server_region"]
            server_type = benchmark["llm_server_type"]
            server_config = benchmark["llm_server_config"] or {}

            # Create and deploy client class
            deploy_benchmark_runner_cls(region)

            # Create and deploy server class if it hasn't already been deployed
            deploy_llm_server_cls(server_type, gpu, region, server_config)

        return app.local_entrypoint()(fn)

    return decorator


@local_entrypoint_with_dynamic_classes()
def run_benchmark_suite_cli(
    config_path: str,
    disable_safe_mode: bool = False,
):
    print(app.registered_classes)
    return

    benchmarks, id, version, repeats = load_benchmarks_from_config(config_path)

    print(len(benchmarks))
    return

    run_benchmark_suite.remote(
        benchmarks=benchmarks,
        id=id,
        version=version,
        repeats=repeats,
        disable_safe_mode=disable_safe_mode,
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
            f"   {datasette_url}/stopwatch/-/query?sql=select+*+from+{id.replace('-', '_')}_averaged"
        )
    except Exception:
        print("- Deploy the Datasette UI with:")
        print("   modal deploy -m stopwatch")

    # Provide path to JSONL file
    print("- Download the results JSONL file with:")
    print(f"   modal volume get stopwatch-results {id}.jsonl")
