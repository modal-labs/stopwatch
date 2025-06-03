import modal

from .constants import HOURS, VersionDefaults
from .etl import export_results
from .resources import app, db_volume, results_volume
from .run_benchmark import all_benchmark_runner_classes


DATASETTE_PATH = "/datasette"
DB_PATH = "/db"
MAX_CONCURRENT_BENCHMARKS = 45
MAX_CONSTANT_RATES = 10
MIN_QPS_STEP_SIZE = 0.5
RESULTS_PATH = "/results"
TIMEOUT = 24 * HOURS  # 1 day


benchmark_suite_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("fastapi[standard]", "numpy", "pandas", "SQLAlchemy")
    .add_local_python_source("cli")
)

with benchmark_suite_image.imports():
    from typing import Any, Dict, List, Optional
    import asyncio
    import itertools
    import json
    import os
    import traceback
    import uuid
    import warnings

    import grpclib
    import numpy as np

    from .db import (
        Benchmark,
        RateType,
        benchmark_cls_factory,
        create_all,
        engine,
        session,
    )


def find_function_call(
    call_graph: List[modal.call_graph.InputInfo], function_call_id: str
):
    for input in call_graph:
        if input.function_call_id == function_call_id:
            return input

        if found_input := find_function_call(input.children, function_call_id):
            return found_input

    return None


def get_benchmarks_to_run(benchmarks: List[Dict[str, Any]]):
    for config in benchmarks:
        benchmark_models = (
            session.query(Benchmark)
            .filter_by(**{k: v for k, v in config.items() if k != "group_id"})
            .all()
        )

        assert len(benchmark_models) <= 1, f"Multiple benchmarks found for {config}"
        benchmark = benchmark_models[0] if benchmark_models else None

        if benchmark is not None:
            if benchmark.start_time is not None:
                # TODO: Check errors?
                continue
            elif benchmark.function_call_id is not None:
                fc = modal.FunctionCall.from_id(benchmark.function_call_id)

                try:
                    call_graph = fc.get_call_graph()
                except grpclib.exceptions.GRPCError as e:
                    if e.status == grpclib.const.Status.NOT_FOUND:
                        # The function call ID is invalid, so we need to re-run
                        # the benchmark
                        pass
                    else:
                        raise e
                else:
                    previous_function_call = find_function_call(
                        call_graph, benchmark.function_call_id
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
                            # If the previous function call has already
                            # completed, we should check if the input is valid.
                            fc.get(timeout=0)
                        except modal.exception.OutputExpiredError:
                            # The result has expired, so we need to re-run the
                            # benchmark since its result wasn't previously
                            # saved to the database
                            pass
                        except TimeoutError:
                            # The previous function call is still running
                            # (since we called get with timeout=0), so we can
                            # save its results directly.
                            yield (benchmark.id, fc)
                            continue
                        except Exception as e:
                            # This function call likely crashed
                            raise Exception(
                                f"The previous function call {benchmark.function_call_id} for benchmark {benchmark.id} crashed. You may ignore this error with --ignore-previous-errors"
                            ) from e
                        else:
                            # The previous function call has completed
                            # successfully, so we can save its results
                            # directly.
                            yield (benchmark.id, fc)
                            continue
        else:
            benchmark = Benchmark(**config)
            session.add(benchmark)
            session.commit()
            db_volume.commit()

        yield (benchmark.id, None)


async def run_benchmark(
    benchmark_id: str,
    fc: Optional[modal.FunctionCall],
    semaphore: asyncio.Semaphore,
):
    async with semaphore:
        benchmark = session.query(Benchmark).filter_by(id=benchmark_id).first()

        if fc is None:
            benchmark_cls = all_benchmark_runner_classes[benchmark.client_region]
            print("Starting benchmark with config", benchmark.get_config())
            fc = benchmark_cls().run_benchmark.spawn(**benchmark.get_config())
            benchmark.function_call_id = fc.object_id
            session.commit()
            db_volume.commit()

        try:
            result = await fc.get.aio()
        except modal.exception.RemoteError as e:
            # Happens when the function call is interrupted manually
            warnings.warn(f"WARNING: Function call result could not be retrieved: {e}")
            return
        except modal.exception.FunctionTimeoutError:
            warnings.warn("WARNING: Benchmark timed out")
            return
        except Exception:
            warnings.warn("WARNING: Unexpected error when running benchmark")
            traceback.print_exc()
            return

        print("Saving results for", fc.object_id)

        with open(os.path.join(RESULTS_PATH, f"{benchmark.id}.json"), "w") as f:
            # The full results are saved to disk since they are too big to
            # fit into the database (~20MB per benchmark)
            json.dump(result, f)

        benchmark.save_results(result)
        session.commit()
        db_volume.commit()


async def run_benchmarks_in_parallel(benchmarks: List[Dict[str, Any]]):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BENCHMARKS)
    tasks = []

    for benchmark_id, fc in get_benchmarks_to_run(benchmarks):
        task = asyncio.create_task(run_benchmark(benchmark_id, fc, semaphore))
        tasks.append(task)

    if len(tasks) == 0:
        return

    await asyncio.gather(*tasks)


@app.function(
    image=benchmark_suite_image,
    volumes={
        DB_PATH: db_volume,
        RESULTS_PATH: results_volume,
    },
    max_containers=1,
    scaledown_window=2,
    timeout=TIMEOUT,
)
@modal.concurrent(max_inputs=1)
async def run_benchmark_suite(
    benchmarks: List[Dict[str, Any]],
    id: str,
    version: int = 1,
    repeats: int = 1,
    disable_safe_mode: bool = False,
):
    db_volume.reload()
    SuiteBenchmark = benchmark_cls_factory(table_name=id.replace("-", "_"))
    SuiteAveragedBenchmark = benchmark_cls_factory(
        table_name=id.replace("-", "_") + "_averaged"
    )
    create_all()

    # STEP 0: Validate benchmarks
    for benchmark in benchmarks:
        for key in [
            "llm_server_type",
            "model",
            "data",
            "gpu",
            "server_region",
            "client_region",
        ]:
            if benchmark.get(key) is None:
                raise ValueError(f"Benchmark {benchmark} has no {key}")

        if "llm_server_config" not in benchmark:
            benchmark["llm_server_config"] = {}

        if "client_config" not in benchmark:
            benchmark["client_config"] = {}

        benchmark["group_id"] = str(uuid.uuid4())[:8]
        benchmark["version_metadata"] = {
            "guidellm": VersionDefaults.GUIDELLM,
            benchmark["llm_server_type"]: benchmark["llm_server_config"].get(
                "version", VersionDefaults.LLM_SERVERS[benchmark["llm_server_type"]]
            ),
            "suite": version,
        }

    # STEP 0.75: Run synchronous benchmarks without repeats to ensure that all
    # benchmark configurations are working.
    if not disable_safe_mode:
        await run_benchmarks_in_parallel(
            [
                {
                    **benchmark,
                    "rate_type": RateType.SYNCHRONOUS.value,
                    "repeat_index": 0,
                }
                for benchmark in benchmarks
            ]
        )

    # STEP 1: Run synchronous and throughput benchmarks
    # TODO: If repeat_index is increased, all of the constant rate benchmarks
    # need to be re-run with the new synchronous and throughput rates in mind
    await run_benchmarks_in_parallel(
        [
            {
                **benchmark,
                "rate_type": rate_type.value,
                "repeat_index": repeat_index,
            }
            for benchmark, rate_type, repeat_index in itertools.product(
                benchmarks,
                [RateType.SYNCHRONOUS, RateType.THROUGHPUT],
                range(repeats),
            )
            if disable_safe_mode
            or session.query(Benchmark)
            .filter_by(
                **{k: v for k, v in benchmark.items() if k != "group_id"},
                rate_type=RateType.SYNCHRONOUS.value,
            )
            .filter(Benchmark.completed_request_rate.is_not(None))
            .count()
            == 1
        ]
    )

    # STEP 2: Run benchmarks at constant rates
    benchmarks_to_run = []
    skipped_benchmark_indices = set()

    for i, benchmark in enumerate(benchmarks):
        synchronous_benchmarks = (
            session.query(Benchmark)
            .filter_by(
                **{k: v for k, v in benchmark.items() if k != "group_id"},
                rate_type=RateType.SYNCHRONOUS.value,
            )
            .filter(
                Benchmark.completed_request_rate.is_not(None)
                & (Benchmark.repeat_index <= repeats)
            )
            .all()
        )
        throughput_benchmarks = (
            session.query(Benchmark)
            .filter_by(
                **{k: v for k, v in benchmark.items() if k != "group_id"},
                rate_type=RateType.THROUGHPUT.value,
            )
            .filter(
                Benchmark.completed_request_rate.is_not(None)
                & (Benchmark.repeat_index <= repeats)
            )
            .all()
        )

        if (
            len(synchronous_benchmarks) < repeats
            or len(throughput_benchmarks) < repeats
        ):
            warnings.warn(
                f"WARNING: Expected {repeats} synchronous and throughput benchmarks, but got {len(synchronous_benchmarks)} and {len(throughput_benchmarks)} for config {benchmark}. Skipping constant rate benchmarks for this config."
            )
            skipped_benchmark_indices.add(i)
            continue

        min_rate = np.mean([x.completed_request_rate for x in synchronous_benchmarks])
        max_rate = np.mean([x.completed_request_rate for x in throughput_benchmarks])

        if min_rate >= max_rate:
            # This generally happens when the model is extremely large and the
            # server can't handle the load well during a throughput test,
            # resulting in 0 or 1 successful requests.
            warnings.warn(
                f"WARNING: Synchronous rate ({min_rate}) is greater than throughput rate ({max_rate}) with config {benchmark}"
            )
            continue

        # By default, run 10 constant-rate runs per benchmark. However, if the
        # QPS step size between constant rates is less than 0.5, run fewer than
        # 10 constant-rate runs.
        for num_constant_rates in range(MAX_CONSTANT_RATES, -1, -1):
            if num_constant_rates == 0:
                constant_rates = []
                break

            constant_rates = np.linspace(min_rate, max_rate, num_constant_rates + 1)[1:]
            qps_step_size = (max_rate - min_rate) / num_constant_rates

            if qps_step_size >= MIN_QPS_STEP_SIZE:
                break

        for rate, repeat_index in itertools.product(constant_rates, range(repeats)):
            benchmarks_to_run.append(
                {
                    **benchmark,
                    "rate_type": RateType.CONSTANT.value,
                    "rate": rate,
                    "repeat_index": repeat_index,
                }
            )

    await run_benchmarks_in_parallel(benchmarks_to_run)

    # STEP 2.5: Delete existing benchmark results
    for cls in [SuiteBenchmark, SuiteAveragedBenchmark]:
        cls.__table__.drop(engine, checkfirst=True)
        cls.__table__.create(engine)

    # STEP 3: Average the results together. Start by finding the parameters
    # that vary between benchmarks in order to get descriptive group IDs for
    # each averaged benchmark.
    parameters = {}

    for benchmark in benchmarks:
        for k in benchmark:
            if isinstance(benchmark[k], dict) or k == "group_id":
                continue

            if k not in parameters:
                parameters[k] = set()

            parameters[k].add(benchmark[k])

    for k in list(parameters.keys()):
        if len(parameters[k]) == 1:
            del parameters[k]

    # Ensure names/group IDs are unique
    group_ids = set()

    for i, benchmark in enumerate(benchmarks):
        base_group_id = (
            "_".join([str(benchmark[k]) for k in sorted(parameters)])
            if len(parameters) > 0
            else "benchmark"
        )
        group_id = base_group_id

        j = 0
        while group_id in group_ids:
            j += 1
            group_id = f"{base_group_id}_{j}"

        group_ids.add(group_id)
        benchmarks[i]["group_id"] = group_id

    for i, benchmark in enumerate(benchmarks):
        if i in skipped_benchmark_indices:
            continue

        data_config = {
            k: int(v)
            for param in benchmark["data"].split(",")
            for k, v in [param.split("=")]
            if hasattr(Benchmark, k)
        }

        benchmark_models = (
            session.query(Benchmark)
            .filter_by(**{k: v for k, v in benchmark.items() if k != "group_id"})
            .all()
        )
        benchmark_rates = set(
            (benchmark.rate_type, benchmark.rate) for benchmark in benchmark_models
        )

        # Clone the benchmark models into the SuiteBenchmark table
        non_pk_columns = [
            k
            for k in Benchmark.__table__.columns.keys()
            if k not in Benchmark.__table__.primary_key.columns.keys()
        ]

        for benchmark_model in benchmark_models:
            session.add(
                SuiteBenchmark(
                    **{c: getattr(benchmark_model, c) for c in non_pk_columns}
                )
            )

        # Average benchmarks with the same parameters
        for rate_type, rate in benchmark_rates:
            averaged_benchmark = SuiteAveragedBenchmark(
                **benchmark,
                **data_config,
                rate_type=rate_type,
                rate=rate,
            )

            for key in [
                "duration",
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
                # TODO: Think about what to do with None values (in tpot_median)
                data = [
                    getattr(b, key)
                    for b in benchmark_models
                    if b.rate_type == rate_type
                    and b.rate == rate
                    and getattr(b, key) is not None
                ]

                if len(data) == 0:
                    continue

                setattr(averaged_benchmark, key, np.median(data))

            session.add(averaged_benchmark)
            print(f"Added averaged benchmark {group_id} for {rate_type} {rate}")

    session.commit()
    db_volume.commit()

    # STEP 4: Export results in frontend format
    export_results.local(SuiteAveragedBenchmark)
