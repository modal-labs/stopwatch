from typing import Any, Mapping, Optional
import contextlib
import urllib.parse

from .db import BenchmarkDefaults
from .trtllm_runner import trtllm
from .vllm_runner import vllm


@contextlib.contextmanager
def llm_server(
    llm_server_type: str,
    model: str,
    *,
    llm_server_config: Optional[Mapping[str, Any]] = None,
    gpu: str = BenchmarkDefaults.GPU,
    region: str = BenchmarkDefaults.REGION,
    profile: bool = False,
):
    if profile and llm_server_type != "vllm":
        raise ValueError("Profiling is only supported for vLLM")

    llm_server_kwargs = {
        "model": model,
        "gpu": gpu,
        "region": region,
        "llm_server_config": llm_server_config,
        "profile": profile,
    }

    if llm_server_type == "vllm":
        from .custom_metrics import vllm_monkey_patch

        with vllm(**llm_server_kwargs) as (vllm_url, extra_query):
            extra_query_args = urllib.parse.urlencode(extra_query)
            metrics_url = f"{vllm_url}/metrics?{extra_query_args}"
            vllm_monkey_patch(metrics_url)

            yield (vllm_url, extra_query)
    elif llm_server_type == "trtllm":
        with trtllm(**llm_server_kwargs) as connection_info:
            yield connection_info
    else:
        raise ValueError(f"Invalid LLM server type: {llm_server_type}")
