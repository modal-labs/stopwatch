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

    llm_server_fn = {
        "vllm": vllm,
        "sglang": sglang,
        "tensorrt-llm": tensorrt_llm,
    }

    if llm_server_type not in llm_server_fn:
        raise ValueError(f"Invalid LLM server type: {llm_server_type}")

    with llm_server_fn[llm_server_type](**llm_server_kwargs) as connection_info:
        yield connection_info
