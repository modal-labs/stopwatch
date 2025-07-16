import logging
from collections.abc import Generator
from pathlib import Path

import modal

from .benchmark.dynamic import (
    create_dynamic_benchmark_runner_cls,
)
from .constants import HOURS, VersionDefaults
from .etl import export_results
from .resources import app, db_volume, results_volume

DATASETTE_PATH = "/datasette"
DB_PATH = "/db"
MAX_CONCURRENT_BENCHMARKS = 45
MAX_CONSTANT_RATES = 10
MIN_QPS_STEP_SIZE = 0.5
RESULTS_PATH = "/results"
TIMEOUT = 24 * HOURS  # 1 day

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


benchmark_suite_image = modal.Image.debian_slim(python_version="3.13").pip_install(
    "fastapi[standard]",
    "numpy",
    "pandas",
    "SQLAlchemy",
)

with benchmark_suite_image.imports():
    import asyncio
    import itertools
    import json
    import traceback
    import uuid
    import warnings
    from typing import Any

    import grpclib
    import numpy as np

    from .db import (
        Benchmark,
        RateType,
        benchmark_class_factory,
        create_all,
        engine,
        session,
    )


def find_function_call(
    call_graph: list[modal.call_graph.InputInfo],
    function_call_id: str,
) -> modal.call_graph.InputInfo | None:
    """
    Return the input to a function call with a specific ID out of all function call
    inputs in a call graph.

    :param: call_graph: The call graph to search.
    :param: function_call_id: The ID of the function call to find.
    :return: The input to the function call with the given ID, or None if no such
        function call is found.
    """

    for fc_input in call_graph:
        if fc_input.function_call_id == function_call_id:
            return fc_input

        if found_input := find_function_call(fc_input.children, function_call_id):
            return found_input

    return None


def get_benchmarks_to_run(
    benchmark_configs: list[dict[str, Any]],
) -> Generator[tuple[list[str] | str, str, str, modal.FunctionCall | None]]:
    """
    Yield IDs of benchmarks that need to be run based on the provided benchmark
    configurations. If a benchmark is already running, its function call will be
    provided. If a benchmark has already completed, it will not be yielded.

    :param: benchmark_configs: The benchmark configurations to use.
    :return: A generator of benchmark IDs and function calls that need to be run.
    """

    logger.info(
        "Reviewing if all %d benchmark configs need to be run...",
        len(benchmark_configs),
    )

    for config_i, config in enumerate(benchmark_configs):
        benchmark_ids_to_run = []

        if isinstance(config["rate_type"], str):
            config["rate_type"] = [config["rate_type"]]

        # If no rate is specified, this is a synchronous or throughput benchmark, so
        # run a single benchmark with no rate. Otherwise, run a benchmark for each rate
        # specified.
        rates_to_run = itertools.product(
            config["rate_type"],
            (
                [None]
                if "rate" not in config
                else (
                    config["rate"]
                    if isinstance(config["rate"], list)
                    else [config["rate"]]
                )
            ),
        )

        for rate_type, rate in rates_to_run:
            benchmark_config = {
                **config,
                "rate_type": rate_type,
            }

            if rate is not None:
                benchmark_config["rate"] = rate

            benchmark_records = (
                session.query(Benchmark)
                .filter_by(
                    **{
                        k: v
                        for k, v in benchmark_config.items()
                        if k not in ("group_id", "client_name", "server_url")
                    },
                )
                .all()
            )

            if len(benchmark_records) > 1:
                msg = (
                    f"Multiple benchmarks found for {benchmark_config}: "
                    f"{benchmark_records}"
                )
                raise Exception(msg)

            benchmark_record = benchmark_records[0] if benchmark_records else None

            if benchmark_record is not None:
                logger.info(
                    "Benchmark %d/%d r=%s started, checking...",
                    config_i + 1,
                    len(benchmark_configs),
                    rate_type,
                )

                if benchmark_record.start_time is not None:
                    # TODO(jack): Check errors?
                    continue
                elif benchmark_record.function_call_id is not None:
                    fc = modal.FunctionCall.from_id(benchmark_record.function_call_id)

                    try:
                        call_graph = fc.get_call_graph()
                    except grpclib.exceptions.GRPCError as e:
                        if e.status == grpclib.const.Status.NOT_FOUND:
                            # The function call ID is invalid, so we need to re-run the
                            # benchmark
                            pass
                        else:
                            raise
                    else:
                        previous_function_call = find_function_call(
                            call_graph,
                            benchmark_record.function_call_id,
                        )

                        if (
                            previous_function_call is not None
                            and previous_function_call.status
                            in [
                                modal.call_graph.InputStatus.PENDING,
                                modal.call_graph.InputStatus.SUCCESS,
                            ]
                        ):
                            try:
                                # If the previous function call has already completed,
                                # we should check if the input is valid.
                                fc.get(timeout=0)
                            except modal.exception.OutputExpiredError:
                                # The result has expired, so we need to re-run the
                                # benchmark since its result wasn't previously saved to
                                # the database.
                                logger.info(
                                    "Benchmark %d/%d result expired, re-running",
                                    config_i + 1,
                                    len(benchmark_configs),
                                )
                            except TimeoutError:
                                # The previous function call is still running (since we
                                # called get with timeout=0), so we can save its results
                                # directly.
                                logger.info(
                                    "Benchmark %d/%d is still running",
                                    config_i + 1,
                                    len(benchmark_configs),
                                )
                                yield (
                                    benchmark_record.id,
                                    config["client_name"],
                                    config["server_url"],
                                    fc,
                                )
                                continue
                            except Exception as e:
                                # This function call likely crashed
                                msg = (
                                    "The previous function call "
                                    f"{benchmark_record.function_call_id} for benchmark"
                                    f" {benchmark_record.id} crashed. You may ignore"
                                    " this error with --ignore-previous-errors",
                                )
                                raise Exception(msg) from e
                            else:
                                # The previous function call has completed successfully,
                                # so we can save its results directly.
                                logger.info(
                                    "Benchmark %d/%d already completed",
                                    config_i + 1,
                                    len(benchmark_configs),
                                )
                                yield (
                                    benchmark_record.id,
                                    config["client_name"],
                                    config["server_url"],
                                    fc,
                                )
                                continue
            else:
                logger.info(
                    "Benchmark %d/%d record created",
                    config_i + 1,
                    len(benchmark_configs),
                )
                benchmark_record = Benchmark(
                    **{
                        k: v
                        for k, v in benchmark_config.items()
                        if k not in ("client_name", "server_url")
                    },
                )
                session.add(benchmark_record)
                session.commit()
                db_volume.commit()

            benchmark_ids_to_run.append(benchmark_record.id)

        # If no benchmark IDs were added to the list, don't yield anything for this
        # config.
        if len(benchmark_ids_to_run) > 0:
            yield (
                benchmark_ids_to_run,
                config["client_name"],
                config["server_url"],
                None,
            )


async def run_benchmarks(
    benchmark_ids: str | list[str],
    client_name: str,
    server_url: str,
    fc: modal.FunctionCall | None,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    Run a benchmark or set of benchmarks, wait for the result(s), and save the
    result(s) to the database.

    :param: benchmark_ids: The database ID(s) of the benchmark(s) to run.
    :param: client_name: The name of the client class to use.
    :param: fc: If this benchmark is already running, the function call that it is
        running on.
    :param: semaphore: A semaphore to limit the number of benchmarks that can run
        concurrently.
    """

    if not isinstance(benchmark_ids, list):
        benchmark_ids = [benchmark_ids]

    async with semaphore:
        benchmark_records = (
            session.query(Benchmark).filter(Benchmark.id.in_(benchmark_ids)).all()
        )

        if len(benchmark_records) == 0:
            msg = f"No benchmark records found for {benchmark_ids}"
            raise Exception(msg)

        rate_types = []
        rates = []

        for benchmark_record in benchmark_records:
            if benchmark_record.client_region != benchmark_records[0].client_region:
                msg = (
                    f"All benchmarks must have the same client region: "
                    f"{benchmark_record.client_region} != "
                    f"{benchmark_records[0].client_region}"
                )
                raise Exception(msg)

            if benchmark_record.rate_type not in rate_types:
                rate_types.append(benchmark_record.rate_type)

            if benchmark_record.rate not in rates:
                rates.append(benchmark_record.rate)

        if len(rate_types) != 1 and len(rates) != 1:
            msg = (
                "All benchmarks must have either the same rate type or the same rate: "
                f"{rate_types} vs. {rates}"
            )
            raise Exception(msg)

        if fc is None:
            config = {
                **benchmark_records[0].get_config(),
                "rate_type": rate_types,
                "rate": rates,
            }

            logger.info("Starting benchmarks with config %s", config)

            # Re-create and hydrate the class that was created during the startup
            # process of the CLI
            benchmark_cls = create_dynamic_benchmark_runner_cls(client_name).hydrate(
                modal.Client.from_env(),
            )

            # Run the benchmark
            fc = benchmark_cls().run_benchmark.spawn(
                endpoint=f"{server_url}/v1",
                caller_id=uuid.uuid4().hex,
                **config,
            )

            for benchmark_record in benchmark_records:
                benchmark_record.function_call_id = fc.object_id

            session.commit()
            db_volume.commit()

        try:
            fc_result = await fc.get.aio()
        except modal.exception.RemoteError as e:
            # Happens when the function call is interrupted manually
            warnings.warn(
                f"WARNING: Function call result could not be retrieved: {e}",
                stacklevel=2,
            )
            return
        except modal.exception.FunctionTimeoutError:
            warnings.warn("WARNING: Benchmark timed out", stacklevel=2)
            return
        except Exception:  # noqa: BLE001
            warnings.warn(
                "WARNING: Unexpected error when running benchmark",
                stacklevel=2,
            )
            traceback.print_exc()
            return

        logger.info("Saving results for %s", fc.object_id)

        with (Path(RESULTS_PATH) / f"{fc.object_id}.json").open("w") as f:
            # The full results are saved to disk since they are too big to fit in the
            # database (~20MB per benchmark run)
            json.dump(fc_result, f)

        db_volume.commit()

        for benchmark_record in benchmark_records:
            benchmark_record.save_results(
                next(
                    r
                    for r in fc_result
                    if r["rate_type"] == benchmark_record.rate_type
                    and r["rate"] == benchmark_record.rate
                )["results"],
            )

        session.commit()


async def run_benchmarks_in_parallel(benchmark_configs: list[dict[str, Any]]) -> None:
    """
    Given a list of benchmark configurations, identify which benchmarks need to be run,
    run them in parallel, and wait until all benchmarks have completed.

    :param: benchmark_configs: The benchmark configurations to run.
    """

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BENCHMARKS)
    tasks = []

    for benchmark_ids, client_name, server_url, fc in get_benchmarks_to_run(
        benchmark_configs,
    ):
        task = asyncio.create_task(
            run_benchmarks(benchmark_ids, client_name, server_url, fc, semaphore),
        )
        tasks.append(task)

        # Yield control to the event loop to allow the task to be scheduled
        await asyncio.sleep(0.01)

    if len(tasks) == 0:
        return

    await asyncio.gather(*tasks)


@app.function(
    image=benchmark_suite_image,
    volumes={
        DB_PATH: db_volume,
        RESULTS_PATH: results_volume,
    },
    cpu=2,
    memory=1 * 1024,
    max_containers=1,
    scaledown_window=2,
    timeout=TIMEOUT,
)
@modal.concurrent(max_inputs=1)
async def run_benchmark_suite(
    benchmarks: list[tuple[dict[str, Any], list[str], list[str]]],
    suite_id: str,
    *,
    version: int = 1,
    repeats: int = 1,
    fast_mode: bool = False,
) -> None:
    """
    Run a suite of benchmarks.

    :param: benchmarks: A list of benchmarks to run. Each item in this list is a tuple
        with three items: the benchmark configuration, the name of the client class to
        use, and the name of the LLM server class to use.
    :param: suite_id: The ID of the benchmark suite.
    :param: version: The version of the benchmark suite.
    :param: repeats: The number of times to repeat each benchmark configuration.
    :param: fast_mode: Whether to run the benchmark suite in fast mode, which will run
        more benchmarks in parallel, incurring extra costs since LLM servers are not
        reused between benchmarks. Disabled by default.
    """

    logger.info(
        "Running %d configs with benchmark suite id %s",
        len(benchmarks),
        suite_id,
    )

    db_volume.reload()
    SuiteBenchmark = benchmark_class_factory(  # noqa: N806
        table_name=suite_id.replace("-", "_"),
    )
    SuiteAveragedBenchmark = benchmark_class_factory(  # noqa: N806
        table_name=suite_id.replace("-", "_") + "_averaged",
    )

    create_all()

    # STEP 0: Validate benchmarks
    logger.info("Validating benchmarks...")

    for benchmark_config, _, _ in benchmarks:
        for key in [
            "llm_server_type",
            "model",
            "data",
            "gpu",
            "server_region",
            "client_region",
        ]:
            if benchmark_config.get(key) is None:
                msg = f"Benchmark {benchmark_config} has no {key}"
                raise Exception(msg)

        if "llm_server_config" not in benchmark_config:
            benchmark_config["llm_server_config"] = {}

        if "client_config" not in benchmark_config:
            benchmark_config["client_config"] = {}

        benchmark_config["group_id"] = uuid.uuid4().hex[:8]
        benchmark_config["version_metadata"] = {
            "guidellm": VersionDefaults.GUIDELLM,
            benchmark_config["llm_server_type"]: benchmark_config[
                "llm_server_config"
            ].get(
                "version",
                VersionDefaults.LLM_SERVERS[benchmark_config["llm_server_type"]],
            ),
            "suite": version,
        }

    # STEP 1: Run synchronous and throughput benchmarks
    logger.info("Running synchronous and throughput benchmarks...")
    # TODO(jack): If repeat_index is increased, all of the constant rate benchmarks need
    # to be re-run with the new synchronous and throughput rates in mind
    await run_benchmarks_in_parallel(
        [
            {
                **benchmark_config,
                "client_name": benchmark_client_names[repeat_index],
                "server_url": benchmark_server_urls[repeat_index],
                "rate_type": rate_type,
                "repeat_index": repeat_index,
            }
            for (
                benchmark_config,
                benchmark_client_names,
                benchmark_server_urls,
            ), rate_type, repeat_index in itertools.product(
                benchmarks,
                (
                    # If fast_mode is enabled, run synchronous and throughput
                    # benchmarks in parallel to save time. Otherwise, run them
                    # sequentially on a single container to save cost.
                    [RateType.SYNCHRONOUS.value, RateType.THROUGHPUT.value]
                    if fast_mode
                    else [[RateType.SYNCHRONOUS.value, RateType.THROUGHPUT.value]]
                ),
                range(repeats),
            )
        ],
    )

    # STEP 2: Run benchmarks at constant rates
    logger.info("Running benchmarks at constant rates...")
    benchmarks_to_run = []
    skipped_benchmark_indices = set()

    for i, (
        benchmark_config,
        benchmark_client_names,
        benchmark_server_urls,
    ) in enumerate(benchmarks):
        synchronous_benchmark_records = (
            session.query(Benchmark)
            .filter_by(
                **{k: v for k, v in benchmark_config.items() if k != "group_id"},
                rate_type=RateType.SYNCHRONOUS.value,
            )
            .filter(
                Benchmark.completed_request_rate.is_not(None)
                & (Benchmark.repeat_index <= repeats),
            )
            .all()
        )
        throughput_benchmark_records = (
            session.query(Benchmark)
            .filter_by(
                **{k: v for k, v in benchmark_config.items() if k != "group_id"},
                rate_type=RateType.THROUGHPUT.value,
            )
            .filter(
                Benchmark.completed_request_rate.is_not(None)
                & (Benchmark.repeat_index <= repeats),
            )
            .all()
        )

        if (
            len(synchronous_benchmark_records) < repeats
            or len(throughput_benchmark_records) < repeats
        ):
            warnings.warn(
                f"WARNING: Expected {repeats} synchronous and throughput benchmarks, "
                f"but got {len(synchronous_benchmark_records)} and "
                f"{len(throughput_benchmark_records)} for config {benchmark_config}. "
                "Skipping constant rate benchmarks for this config.",
                stacklevel=2,
            )
            skipped_benchmark_indices.add(i)
            continue

        min_rate = np.mean(
            [x.completed_request_rate for x in synchronous_benchmark_records],
        )
        max_rate = np.mean(
            [x.completed_request_rate for x in throughput_benchmark_records],
        )

        if min_rate >= max_rate:
            # This generally happens when the model is extremely large and the server
            # can't handle the load well during a throughput test, resulting in 0 or 1
            # successful requests.
            warnings.warn(
                f"WARNING: Synchronous rate ({min_rate}) is greater than throughput "
                f"rate ({max_rate}) with config {benchmark_config}",
                stacklevel=2,
            )
            continue

        # By default, run 10 constant-rate runs per benchmark. However, if the QPS step
        # size between constant rates is less than 0.5, run fewer than 10 constant-rate
        # runs.
        for num_constant_rates in range(MAX_CONSTANT_RATES, -1, -1):
            if num_constant_rates == 0:
                constant_rates = []
                break

            constant_rates = np.linspace(min_rate, max_rate, num_constant_rates + 1)[
                1:
            ].tolist()
            qps_step_size = (max_rate - min_rate) / num_constant_rates

            if qps_step_size >= MIN_QPS_STEP_SIZE:
                break

        for repeat_index in range(repeats):
            # If fast_mode is enabled, run all constant-rate benchmarks in parallel to
            # save time. Otherwise, run them sequentially on a single container to save
            # cost.
            benchmarks_to_run.extend(
                [
                    {
                        **benchmark_config,
                        "client_name": benchmark_client_names[repeat_index],
                        "server_url": benchmark_server_urls[repeat_index],
                        "rate_type": RateType.CONSTANT.value,
                        "rate": rate,
                        "repeat_index": repeat_index,
                    }
                    for rate in (constant_rates if fast_mode else [constant_rates])
                ],
            )

    await run_benchmarks_in_parallel(benchmarks_to_run)

    # STEP 2.5: Delete existing benchmark results
    logger.info("Deleting existing benchmark results...")

    for cls in [SuiteBenchmark, SuiteAveragedBenchmark]:
        cls.__table__.drop(engine, checkfirst=True)
        cls.__table__.create(engine)

    # STEP 3: Average the results together. Start by finding the parameters that vary
    # between benchmarks in order to get descriptive group IDs for each averaged
    # benchmark.
    logger.info("Averaging benchmark results...")

    parameters = {}

    for benchmark_config, _, _ in benchmarks:
        for k in benchmark_config:
            if isinstance(benchmark_config[k], dict) or k == "group_id":
                continue

            if k not in parameters:
                parameters[k] = set()

            parameters[k].add(benchmark_config[k])

    for k in list(parameters.keys()):
        if len(parameters[k]) == 1:
            del parameters[k]

    # Ensure names/group IDs are unique
    group_ids = set()

    for i, (benchmark_config, _, _) in enumerate(benchmarks):
        base_group_id = (
            "_".join([str(benchmark_config[k]) for k in sorted(parameters)])
            if len(parameters) > 0
            else "benchmark"
        )
        group_id = base_group_id

        j = 0
        while group_id in group_ids:
            j += 1
            group_id = f"{base_group_id}_{j}"

        group_ids.add(group_id)
        benchmarks[i][0]["group_id"] = group_id

    for i, (benchmark_config, _, _) in enumerate(benchmarks):
        if i in skipped_benchmark_indices:
            continue

        data_config = {
            k: int(v)
            for param in benchmark_config["data"].split(",")
            for k, v in [param.split("=")]
            if hasattr(Benchmark, k)
        }

        benchmark_records = (
            session.query(Benchmark)
            .filter_by(**{k: v for k, v in benchmark_config.items() if k != "group_id"})
            .all()
        )
        benchmark_rates = {
            (benchmark_record.rate_type, benchmark_record.rate)
            for benchmark_record in benchmark_records
        }

        # Clone the benchmark models into the SuiteBenchmark table
        non_pk_columns = [
            k
            for k in Benchmark.__table__.columns.keys()  # noqa: SIM118
            if k not in Benchmark.__table__.primary_key.columns.keys()  # noqa: SIM118
        ]

        for benchmark_record in benchmark_records:
            session.add(
                SuiteBenchmark(
                    **{c: getattr(benchmark_record, c) for c in non_pk_columns},
                ),
            )

        # Average benchmarks with the same parameters
        for rate_type, rate in benchmark_rates:
            averaged_benchmark = SuiteAveragedBenchmark(
                **benchmark_config,
                **data_config,
                rate_type=rate_type,
                rate=rate,
            )

            for key in [
                "duration",
                "queue_duration",
                "cold_start_duration",
                "completed_request_count",
                "completed_request_rate",
                "tpot_median",
                "kv_cache_usage_mean",
                *[
                    f"{statistic}_{percentile}"
                    for statistic, percentile in itertools.product(
                        ["itl", "ttft", "ttlt"],
                        ["mean", "p50", "p90", "p95", "p99"],
                    )
                ],
            ]:
                # TODO(jack): Think about what to do with None values (in tpot_median)
                data = [
                    getattr(b, key)
                    for b in benchmark_records
                    if b.rate_type == rate_type
                    and b.rate == rate
                    and getattr(b, key) is not None
                ]

                if len(data) == 0:
                    continue

                setattr(averaged_benchmark, key, np.median(data))

            session.add(averaged_benchmark)
            logger.info(
                "Added averaged benchmark %s for %s %s",
                group_id,
                rate_type,
                rate,
            )

    session.commit()
    db_volume.commit()

    # STEP 4: Export results in frontend format
    logger.info("Exporting results to frontend format...")
    export_results.local(SuiteAveragedBenchmark)
