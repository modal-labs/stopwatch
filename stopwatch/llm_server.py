from typing import Any, Mapping, Optional
import contextlib

from .sglang_runner import sglang
from .tensorrt_llm_runner import tensorrt_llm
from .vllm_runner import vllm


@contextlib.contextmanager
def llm_server(
    llm_server_type: str,
    *,
    model: str,
    gpu: str,
    region: str,
    server_config: Optional[Mapping[str, Any]] = None,
    profile: bool = False,
):
    if profile and llm_server_type != "vllm":
        raise ValueError("Profiling is only supported for vLLM")

    llm_server_kwargs = {
        "model": model,
        "gpu": gpu,
        "region": region,
        "server_config": server_config,
        "profile": profile,
    }

    if llm_server_type == "vllm":
        with vllm(**llm_server_kwargs) as (vllm_url, extra_query):
            yield (vllm_url, extra_query)
    elif llm_server_type == "sglang":
        with sglang(**llm_server_kwargs) as (sglang_url, extra_query):
            yield (sglang_url, extra_query)
    elif llm_server_type == "tensorrt-llm":
        with tensorrt_llm(**llm_server_kwargs) as connection_info:
            yield connection_info
    else:
        raise ValueError(f"Invalid LLM server type: {llm_server_type}")
