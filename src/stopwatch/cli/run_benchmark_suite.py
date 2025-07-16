import fnmatch
import itertools
import uuid
from pathlib import Path
from typing import Any

import modal
import yaml

from stopwatch.benchmark.dynamic import create_dynamic_benchmark_runner_cls
from stopwatch.llm_servers.dynamic import create_dynamic_llm_server_cls
from stopwatch.resources import app, db_volume
from stopwatch.run_benchmark_suite import run_benchmark_suite


def build_all_benchmark_configs(
    config_path: Path,
    exclude_instance_types: str | None = None,
) -> tuple[list[dict[str, Any]], str, int, int]:
    """
    Build benchmark configurations by computing the outer product of all listed
    configurations in the YAML config file.

    :param: config_path: The path to the YAML file containing the benchmark
        configurations.
    :param: exclude_instance_types: A comma-separated list of instance types to exclude
        from the benchmark suite. Asterisks are supported, e.g. "H100:*" or "*:8".
    :return: A tuple containing all specified benchmark configurations, the suite ID,
        the suite version, and the number of times the suite should be repeated.
    """

    with config_path.open() as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if "id" not in config:
        msg = "'id' is required in the config"
        raise ValueError(msg)

    benchmark_configs = []

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

            if exclude_instance_types is not None and any(
                fnmatch.fnmatch(benchmark_config["gpu"], exclude_filter)
                for exclude_filter in exclude_instance_types.split(",")
            ):
                continue

            benchmark_configs.append(benchmark_config)

    for file in config.get("files", []):
        file_benchmark_configs, _, _, _ = build_all_benchmark_configs(
            config_path.parent / file,
            exclude_instance_types,
        )
        benchmark_configs.extend(file_benchmark_configs)

    return (
        benchmark_configs,
        config["id"],
        config.get("version", 1),
        config.get("repeats", 1),
    )


def run_benchmark_suite_cli(
    config_path: str,
    *,
    detach: bool = False,
    exclude_instance_types: str | None = None,
    fast_mode: bool = False,
) -> None:
    """
    Run a benchmark suite.

    :param: config_path: The path to the YAML file containing the benchmark
        configurations.
    :param: exclude_instance_types: A comma-separated list of instance types to exclude
        from the benchmark suite. Asterisks are supported, e.g. "H100:*" or "*:8".
    :param: fast_mode: Whether to run the benchmark suite in fast mode, which will run
        more benchmarks in parallel, incurring extra costs since LLM servers are not
        reused between benchmarks. Disabled by default.
    """

    benchmark_configs, suite_id, version, repeats = build_all_benchmark_configs(
        Path(config_path),
        exclude_instance_types,
    )

    if len(benchmark_configs) == 0:
        print("No benchmarks to run")
        return

    benchmark_names = [
        [uuid.uuid4().hex[:8] for _ in range(repeats)]
        for _ in range(len(benchmark_configs))
    ]

    benchmark_clients = [
        [
            create_dynamic_benchmark_runner_cls(
                benchmark_names[i][j],
                benchmark_config.get("client_region"),
            )
            for j in range(repeats)
        ]
        for i, benchmark_config in enumerate(benchmark_configs)
    ]

    with db_volume.batch_upload(force=True) as batch:
        benchmark_servers = [
            [
                create_dynamic_llm_server_cls(
                    benchmark_names[i][j],
                    benchmark_config["model"],
                    gpu=benchmark_config["gpu"],
                    llm_server_type=benchmark_config["llm_server_type"],
                    region=benchmark_config.get("server_region"),
                    llm_server_config=benchmark_config.get("llm_server_config", {}),
                    batch=batch,
                    parametrized_fn=True,
                )
                for j in range(repeats)
            ]
            for i, benchmark_config in enumerate(benchmark_configs)
        ]

    with modal.enable_output(), app.run(detach=detach):
        benchmark_client_names = [
            [client_cls.__name__ for client_cls in client_classes]
            for client_classes in benchmark_clients
        ]

        benchmark_server_urls = [
            [server_cls().start.get_web_url() for server_cls in server_classes]
            for server_classes in benchmark_servers
        ]

        run_benchmark_suite.remote(
            benchmarks=list(
                zip(
                    benchmark_configs,
                    benchmark_client_names,
                    benchmark_server_urls,
                    strict=True,
                ),
            ),
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
                "'throughput'",
            )
        except Exception:  # noqa: BLE001
            print("- Deploy the Datasette UI with:")
            print("   modal deploy -m stopwatch")

        # Provide path to JSONL file
        print("- Download the results JSONL file with:")
        print(f"   modal volume get stopwatch-results {suite_id}.jsonl")
