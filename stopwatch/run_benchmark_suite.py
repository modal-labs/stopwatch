import modal

from .db import (
    Benchmark,
    DEFAULT_LLM_SERVER_CONFIGS,
    benchmark_cls_factory,
    create_all,
    engine,
    session,
)
from .resources import app, db_volume, results_volume
from .run_benchmark import all_benchmark_runner_classes


DATASETTE_PATH = "/datasette"
DB_PATH = "/db"
MAX_CONCURRENT_BENCHMARKS = 45
MAX_CONSTANT_RATES = 10
MIN_QPS_STEP_SIZE = 0.5
RESULTS_PATH = "/results"
TIMEOUT = 12 * 60 * 60  # 12 hours


benchmark_suite_image = modal.Image.debian_slim(python_version="3.13").pip_install(
    "numpy", "SQLAlchemy"
)

with benchmark_suite_image.imports():
    from typing import Any, Dict, List, Optional
    import asyncio
    import itertools
    import json
    import os
    import uuid

    import grpclib
    import numpy as np


def find_function_call(
    call_graph: List[modal.call_graph.InputInfo], function_call_id: str
):
    for input in call_graph:
        if input.function_call_id == function_call_id:
            return input

        if found_input := find_function_call(input.children, function_call_id):
            return found_input

    return None


def get_benchmarks_to_run(
    benchmarks: List[Dict[str, Any]], ignore_previous_errors: bool = False
):
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
                            # This function call likely crashed, so we should
                            # only re-run it if ignore_previous_errors is set
                            # to true.
                            if ignore_previous_errors:
                                print(
                                    "WARNING: Unexpected error when checking previous function call",
                                    e,
                                )
                                pass
                            else:
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
            print("WARNING: Function call result could not be retrieved:", e)
            return
        except modal.exception.FunctionTimeoutError:
            print("Benchmark timed out")
            return

        if len(result["results"]) == 0:
            # This happens when the benchmark is run with invald parameters
            # e.g. asking the model to generate more tokens than its
            # maximum context size. When this happens, requests made to the
            # vLLM runner return a 400 error, and no results are saved.

            # TODO: Return an error when 400 errors are encountered without
            # crashing the run_benchmark_suite function.

            print("No results for", fc.object_id)
            benchmark.function_call_id = None
            session.commit()
            db_volume.commit()
            return

        print("Saving results for", fc.object_id)

        with open(os.path.join(RESULTS_PATH, f"{benchmark.id}.json"), "w") as f:
            # The full results are saved to disk since they are too big to
            # fit into the database (~20MB per benchmark)
            json.dump(result, f)

        benchmark.save_results(result["results"], result.get("vllm_metrics", None))
        session.commit()
        db_volume.commit()


async def run_benchmarks_in_parallel(
    benchmarks: List[Dict[str, Any]],
    dry_run: bool = False,
    ignore_previous_errors: bool = False,
):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BENCHMARKS)

    while True:
        tasks = []

        for benchmark_id, fc in get_benchmarks_to_run(
            benchmarks, ignore_previous_errors=ignore_previous_errors
        ):
            if dry_run:
                benchmark = session.query(Benchmark).filter_by(id=benchmark_id).first()
                print("Would run benchmark with config", benchmark.get_config())
                continue

            task = asyncio.create_task(run_benchmark(benchmark_id, fc, semaphore))
            tasks.append(task)

        if dry_run or len(tasks) == 0:
            break

        await asyncio.gather(*tasks)


@app.function(
    image=benchmark_suite_image,
    volumes={
        DB_PATH: db_volume,
        RESULTS_PATH: results_volume,
    },
    max_containers=1,
    scaledown_window=2,
    allow_concurrent_inputs=1,
    timeout=TIMEOUT,
)
async def run_benchmark_suite(
    benchmarks: List[Dict[str, Any]],
    id: str,
    repeats: int = 1,
    dry_run: bool = False,
    ignore_previous_errors: bool = False,
    recompute: bool = False,
):
    db_volume.reload()
    AveragedBenchmark = benchmark_cls_factory(table_name=id.replace("-", "_"))
    create_all()

    # STEP 0: Validate benchmarks
    for i, benchmark in enumerate(benchmarks):
        for key in [
            "llm_server_type",
            "model",
            "data",
            "gpu",
            "server_region",
            "client_region",
        ]:
            if benchmark.get(key) is None:
                raise ValueError(f"Benchmark {i} has no {key}")

        if "llm_server_config" not in benchmark:
            benchmark["llm_server_config"] = DEFAULT_LLM_SERVER_CONFIGS[
                benchmark["llm_server_type"]
            ]

        benchmark["group_id"] = str(uuid.uuid4())[:8]

    # STEP 0.5: Delete existing benchmarks if recompute is set to true
    if recompute:
        # Delete existing benchmarks if recompute is set to true
        for benchmark in benchmarks:
            session.query(Benchmark).filter_by(
                **{k: v for k, v in benchmark.items() if k != "group_id"}
            ).delete()

        session.commit()

    # STEP 1: Run synchronous and throughput benchmarks
    # TODO: If repeat_index is increased, all of the constant rate benchmarks
    # need to be re-run with the new synchronous and throughput rates in mind
    await run_benchmarks_in_parallel(
        [
            {
                **benchmark,
                "rate_type": rate_type,
                "repeat_index": repeat_index,
            }
            for benchmark, rate_type, repeat_index in itertools.product(
                benchmarks, ["synchronous", "throughput"], range(repeats)
            )
        ],
        dry_run=dry_run,
        ignore_previous_errors=ignore_previous_errors,
    )

    if dry_run:
        return

    # STEP 2: Run benchmarks at constant rates
    benchmarks_to_run = []

    for benchmark in benchmarks:
        # TODO: Use constants for synchronous and throughput
        synchronous_benchmarks = (
            session.query(Benchmark)
            .filter_by(
                **{k: v for k, v in benchmark.items() if k != "group_id"},
                rate_type="synchronous",
            )
            .all()
        )
        min_rate = np.mean([x.completed_request_rate for x in synchronous_benchmarks])

        throughput_benchmarks = (
            session.query(Benchmark)
            .filter_by(
                **{k: v for k, v in benchmark.items() if k != "group_id"},
                rate_type="throughput",
            )
            .all()
        )
        max_rate = np.mean([x.completed_request_rate for x in throughput_benchmarks])

        assert min_rate < max_rate

        # By default, run 10 constant-rate runs per benchmark. However, if the
        # QPS step size between constant rates is less than 0.5, run fewer than
        # 10 constant-rate runs.
        for num_constant_rates in range(MAX_CONSTANT_RATES, -1, -1):
            constant_rates = np.linspace(min_rate, max_rate, num_constant_rates + 1)[1:]
            qps_step_size = (max_rate - min_rate) / num_constant_rates

            if qps_step_size >= MIN_QPS_STEP_SIZE:
                break

        for rate, repeat_index in itertools.product(constant_rates, range(repeats)):
            benchmarks_to_run.append(
                {
                    **benchmark,
                    "rate_type": "constant",
                    "rate": rate,
                    "repeat_index": repeat_index,
                }
            )

    await run_benchmarks_in_parallel(benchmarks_to_run)

    # STEP 2.5: Delete existing averaged benchmarks
    AveragedBenchmark.__table__.drop(engine, checkfirst=True)
    AveragedBenchmark.__table__.create(engine)

    # STEP 3: Average the results together. Start by finding the parameters
    # that vary between benchmarks in order to get descriptive group IDs for
    # each averaged benchmark.
    parameters = {}

    for benchmark in benchmarks:
        for k in benchmark:
            if type(benchmark[k]) == dict or k == "group_id":
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

    for benchmark in benchmarks:
        benchmark_models = (
            session.query(Benchmark)
            .filter_by(**{k: v for k, v in benchmark.items() if k != "group_id"})
            .all()
        )
        benchmark_rates = set(
            (benchmark.rate_type, benchmark.rate) for benchmark in benchmark_models
        )

        for rate_type, rate in benchmark_rates:
            averaged_benchmark = AveragedBenchmark(
                **benchmark,
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


@app.function(
    image=benchmark_suite_image,
    volumes={
        DB_PATH: db_volume,
        RESULTS_PATH: results_volume,
    },
    timeout=TIMEOUT,
)
def reload_results():
    """Full results are saved to disk, while only select statistics are saved
    to the database. This function saves each benchmark's full results to the
    database. This can be useful if the save_results function is updated to
    save new statistics."""

    benchmark_models = session.query(Benchmark).all()

    for benchmark_model in benchmark_models:
        results_path = os.path.join(RESULTS_PATH, f"{benchmark_model.id}.json")

        if not os.path.exists(results_path):
            continue

        with open(results_path, "r") as f:
            results = json.load(f)

        benchmark_model.save_results(
            results["results"], results.get("vllm_metrics", None)
        )
        session.commit()

    db_volume.commit()
