import urllib.parse

import modal

from .benchmark import BenchmarkDefaults, get_benchmark_fingerprint
from .resources import app, hf_secret, results_dict, results_volume
from .vllm_runner import vllm
from .trtllm_runner import trtllm


# MAX_SECONDS_PER_BENCHMARK_RUN = 120  # 2 minutes
MAX_SECONDS_PER_BENCHMARK_RUN = 20  #FIXME # 2 minutes
RESULTS_PATH = "/results"
SCALEDOWN_WINDOW = 5  # 5 seconds
TIMEOUT = 60 * 60  # 1 hour

benchmarking_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("git+https://github.com/neuralmagic/guidellm.git", "prometheus-client")
)

with benchmarking_image.imports():
    from typing import Dict, List
    import os

    from guidellm import generate_benchmark_report

    from .custom_metrics import vllm_monkey_patch


def benchmark_runner_cls(region: str):
    def decorator(cls):
        return app.cls(
            image=benchmarking_image,
            secrets=[hf_secret],
            volumes={RESULTS_PATH: results_volume},
            cpu=4,
            memory=2048,
            scaledown_window=SCALEDOWN_WINDOW,
            timeout=TIMEOUT,
            region=region,
        )(cls)

    return decorator


class BenchmarkRunner:
    @modal.method()
    def run_benchmark(
        self,
        model: str,
        data: str = BenchmarkDefaults.DATA,
        data_type: str = BenchmarkDefaults.DATA_TYPE,
        gpu: str = BenchmarkDefaults.GPU,
        region: str = BenchmarkDefaults.REGION,
        llm_engine_type: str = BenchmarkDefaults.LLM_ENGINE_TYPE, 
        llm_engine_version: str = BenchmarkDefaults.LLM_ENGINE_VERSION, 
        llm_env_vars: Dict[str, str] = BenchmarkDefaults.LLM_ENV_VARS,
        llm_extra_args: List[str] = BenchmarkDefaults.LLM_EXTRA_ARGS,
        repeat_index: int = 0,
    ):
        """Benchmarks a LLM deployment on Modal.

        Args:
            model (str, required): Name of the model to benchmark.
            data (str): The data source to use for benchmarking. Depending on the
                data-type, it should be a path to a data file containing prompts to
                run (ex: data.txt), a HuggingFace dataset name (ex:
                'neuralmagic/LLM_compression_calibration'), or a configuration for
                emulated data (ex: 'prompt_tokens=128,generated_tokens=128').
            data_type (str): The type of data to use, such as 'emulated', 'file',
                or 'transformers'.
            gpu (str): The GPU to use for benchmarking.
            region (str): The region to use for benchmarking.
            llm_engine_type (str): vLLM or TRTLLM
            llm_engine_version (str): LLM version (e.g. 0.7.3)
            llm_env_vars (dict): Environment variables to pass to the LLM server.
            llm_extra_args (list): Extra arguments to pass to the LLM server.
        """

        caller_id = modal.current_function_call_id()
        extra_query = {
            "model": model,
            # Include caller_id in extra_query to ensure that similar benchmark
            # runs are given separate vLLM server instances
            "caller_id": caller_id,
        }

        if len(llm_extra_args) > 0:
            extra_query["extra_llm_args"] = " ".join(llm_extra_args)
        if len(llm_env_vars) > 0:
            extra_query["llm_env_vars"] = " ".join(
                f"{k}={v}" for k, v in llm_env_vars.items()
            )

        engine_name_to_params = {
            "vllm": (vllm, vllm_monkey_patch),
            "trtllm": (trtllm, lambda x: None),
        }


        llm, monkey_patch = engine_name_to_params[llm_engine_type.strip().lower()]

        # Start LLM server in background
        with llm(
            llm_engine_version=llm_engine_version,
            extra_query=extra_query,
            gpu=gpu,
            region=region,
        ) as llm_url:
            extra_query_args = urllib.parse.urlencode(extra_query)
            metrics_url = f"{llm_url}/metrics?{extra_query_args}"
            monkey_patch(metrics_url)

            # Run benchmark with guidellm
            generate_benchmark_report(
                target=f"{llm_url}/v1",
                backend="openai_server",
                model=model,
                data=data,
                data_type=data_type,
                tokenizer=None,
                rate_type="sweep",
                rate=None,
                max_seconds=MAX_SECONDS_PER_BENCHMARK_RUN,
                max_requests=None,
                output_path=os.path.join(RESULTS_PATH, f"{caller_id}.json"),
                cont_refresh_table=False,
                backend_kwargs={
                    "extra_query": extra_query,
                },
            )
            results_volume.commit()

        # Cache results path
        fingerprint = get_benchmark_fingerprint(
            model=model,
            data=data,
            data_type=data_type,
            gpu=gpu,
            region=region,
            llm_engine_type=llm_engine_type,
            llm_engine_version=llm_engine_version,
            llm_env_vars=llm_env_vars,
            llm_extra_args=llm_extra_args,
            repeat_index=repeat_index,
        )
        results_dict[fingerprint] = caller_id


@benchmark_runner_cls(region="us-ashburn-1")
class BenchmarkRunner_OCI_USASHBURN1(BenchmarkRunner):
    pass


@benchmark_runner_cls(region="us-east-1")
class BenchmarkRunner_AWS_USEAST1(BenchmarkRunner):
    pass


@benchmark_runner_cls(region="us-east4")
class BenchmarkRunner_GCP_USEAST4(BenchmarkRunner):
    pass


all_benchmark_runner_classes = {
    "us-ashburn-1": BenchmarkRunner_OCI_USASHBURN1,
    "us-east-1": BenchmarkRunner_AWS_USEAST1,
    "us-east4": BenchmarkRunner_GCP_USEAST4,
}
