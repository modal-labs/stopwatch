import modal

from .async_utils import as_completed
from .db import (
    Benchmark,
    BenchmarkDefaults,
    benchmark_cls_factory,
    create_all,
    session,
)
from .resources import app, db_volume, results_volume
from .run_benchmark import all_benchmark_runner_classes


DATASETTE_PATH = "/datasette"
DB_PATH = "/db"
MAX_CONCURRENT_BENCHMARKS = 20
NUM_CONSTANT_RATES = 10
RESULTS_PATH = "/results"
TIMEOUT = 12 * 60 * 60  # 12 hours


benchmark_suite_image = modal.Image.debian_slim(python_version="3.13").pip_install(
    "numpy", "SQLAlchemy"
)

with benchmark_suite_image.imports():
    from typing import Any, Dict, List
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


def get_benchmarks_to_run(benchmarks: List[Dict[str, Any]]):
    pending_function_calls = []

    for config in benchmarks:
        if len(pending_function_calls) >= MAX_CONCURRENT_BENCHMARKS:
            break

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
                    if e.code == grpclib.const.Status.NOT_FOUND:
                        # The function call ID is invalid, so we need to re-run
                        # the benchmark
                        pass
                    else:
                        raise e
                else:
                    previous_function_call = find_function_call(
                        call_graph, benchmark.function_call_id
                    )

                    if previous_function_call is not None:
                        if (
                            previous_function_call.status
                            == modal.call_graph.InputStatus.PENDING
                        ):
                            # The previous function call is still  running, so we
                            # don't need to run it again
                            pending_function_calls.append((benchmark.id, fc))
                            continue
                        elif (
                            previous_function_call.status
                            == modal.call_graph.InputStatus.SUCCESS
                        ):
                            try:
                                # The previous function call has already completed,
                                # so we should check if the input is valid
                                fc.get(timeout=0)
                            except modal.exception.OutputExpiredError:
                                # The result has expired, so we need to re-run the
                                # benchmark since its result wasn't previously
                                # saved to the database
                                pass
                            else:
                                pending_function_calls.append((benchmark.id, fc))
                                continue
        else:
            benchmark = Benchmark(**config)
            session.add(benchmark)
            session.commit()

        benchmark_cls = all_benchmark_runner_classes[benchmark.region]
        print("Starting benchmark with config", benchmark.get_config())
        fc = benchmark_cls().run_benchmark.spawn(**benchmark.get_config())

        benchmark.function_call_id = fc.object_id
        session.commit()

        pending_function_calls.append((benchmark.id, fc))

    db_volume.commit()
    return pending_function_calls


def run_benchmarks_in_parallel(benchmarks: List[Dict[str, Any]]):
    while True:
        pending_benchmarks = get_benchmarks_to_run(benchmarks)

        if len(pending_benchmarks) == 0:
            break

        for function_call_id, result in as_completed(
            [fc for _, fc in pending_benchmarks]
        ):
            if isinstance(result, Exception):
                print("Error retrieving result:", result)
                continue

            benchmark_model = (
                session.query(Benchmark)
                .filter_by(function_call_id=function_call_id)
                .first()
            )

            if len(result["results"]) == 0:
                # This happens when the benchmark is run with invald parameters
                # e.g. asking the model to generate more tokens than its
                # maximum context size. When this happens, requests made to the
                # vLLM runner return a 400 error, and no results are saved.

                # TODO: Return an error when 400 errors are encountered without
                # crashing the run_benchmark_suite function.

                print("No results for", function_call_id)
                benchmark_model.function_call_id = None
                session.commit()
                continue

            print("Saving results for", function_call_id)

            with open(
                os.path.join(RESULTS_PATH, f"{benchmark_model.id}.json"), "w"
            ) as f:
                # The full results are saved to disk since they are too big to
                # fit into the database (~20MB per benchmark)
                json.dump(result, f)

            benchmark_model.save_results(
                result["results"], result.get("vllm_metrics", None)
            )
            session.commit()
            db_volume.commit()


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
def run_benchmark_suite(
    benchmarks: List[Dict[str, Any]],
    id: str,
    repeats: int = 1,
    recompute: bool = False,
):
    db_volume.reload()
    AveragedBenchmark = benchmark_cls_factory(table_name=id.replace("-", "_"))
    create_all()

    # STEP -1: Delete existing averaged benchmarks
    session.query(AveragedBenchmark).delete()
    session.commit()

    # STEP 0: Validate benchmarks
    for i, benchmark in enumerate(benchmarks):
        for key in ["model", "data", "llm_server_type"]:
            if benchmark.get(key) is None:
                raise ValueError(f"Benchmark {i} has no {key}")

        for key in ["gpu", "region"]:
            if benchmark.get(key) is None:
                benchmarks[i][key] = getattr(BenchmarkDefaults, key.upper())

        if "llm_server_config" not in benchmark:
            benchmark["llm_server_config"] = BenchmarkDefaults.LLM_SERVER_CONFIGS[
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
    run_benchmarks_in_parallel(
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
    )

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

        constant_rates = np.linspace(min_rate, max_rate, NUM_CONSTANT_RATES + 1)[1:]

        for rate, repeat_index in itertools.product(constant_rates, range(repeats)):
            benchmarks_to_run.append(
                {
                    **benchmark,
                    "rate_type": "constant",
                    "rate": rate,
                    "repeat_index": repeat_index,
                }
            )

    run_benchmarks_in_parallel(benchmarks_to_run)

    # STEP 3: Average the results together
    for benchmark in benchmarks:
        benchmark_models = (
            session.query(Benchmark)
            .filter_by(**{k: v for k, v in benchmark.items() if k != "group_id"})
            .all()
        )
        benchmark_rates = set(
            (benchmark.rate_type, benchmark.rate) for benchmark in benchmark_models
        )
        group_id = str(uuid.uuid4())[:8]

        for rate_type, rate in benchmark_rates:
            averaged_benchmark = AveragedBenchmark(
                **{
                    **benchmark,
                    "group_id": group_id,
                    "rate_type": rate_type,
                    "rate": rate,
                }
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
