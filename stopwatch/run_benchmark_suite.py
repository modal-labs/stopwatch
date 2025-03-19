import modal

from .async_utils import as_completed
from .db import (
    AveragedBenchmark,
    Benchmark,
    BenchmarkDefaults,
    create_all,
    session,
)
from .resources import app, db_volume, results_volume
from .run_benchmark import all_benchmark_runner_classes


DATASETTE_PATH = "/datasette"
DB_PATH = "/db"
MAX_CONCURRENT_BENCHMARKS = 25
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


def get_benchmarks_to_run(benchmarks: List[Dict[str, Any]], recompute: bool = False):
    pending_function_calls = []

    for config in benchmarks:
        if len(pending_function_calls) >= MAX_CONCURRENT_BENCHMARKS:
            break

        benchmark_models = (
            session.query(Benchmark)
            .filter_by(**{k: v for k, v in config.items() if k != "group_id"})
            .all()
        )

        assert len(benchmark_models) <= 1
        benchmark = benchmark_models[0] if benchmark_models else None

        if not recompute and benchmark is not None:
            if benchmark.start_time is not None:
                # TODO: Check errors?
                continue
            else:
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
            elif len(result["results"]) == 0:
                # TODO: Not sure why this happens, but it results in us needing
                # to re-run this benchmark
                print("No results for", function_call_id)
                continue

            print("Saving results for", function_call_id)

            benchmark_model = (
                session.query(Benchmark)
                .filter_by(function_call_id=function_call_id)
                .first()
            )

            with open(
                os.path.join(RESULTS_PATH, f"{benchmark_model.id}.json"), "w"
            ) as f:
                # The full results are saved to disk since they are too big to
                # fit into the database (~20MB per benchmark)
                json.dump(result, f)

            benchmark_model.save_results(result["results"], result["vllm_metrics"])
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
def run_benchmark_suite(
    benchmarks: List[Dict[str, Any]],
    repeats: int = 1,
    recompute: bool = False,
):
    create_all()

    # STEP 0: Validate benchmarks
    for i, benchmark in enumerate(benchmarks):
        for key in ["model", "data"]:
            if benchmark.get(key) is None:
                raise ValueError(f"Benchmark {i} has no {key}")

        for key in [
            "data",
            "gpu",
            "region",
            "vllm_docker_tag",
            "vllm_env_vars",
            "vllm_extra_args",
        ]:
            if benchmark.get(key) is None:
                benchmarks[i][key] = getattr(BenchmarkDefaults, key.upper())

        benchmark["group_id"] = str(uuid.uuid4())[:8]

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
        ]
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
            session.add(
                AveragedBenchmark(
                    [
                        benchmark
                        for benchmark in benchmark_models
                        if benchmark.rate_type == rate_type and benchmark.rate == rate
                    ],
                    group_id=group_id,
                )
            )
            print(f"Added averaged benchmark {group_id} for {rate_type} {rate}")

    session.commit()
    db_volume.commit()
