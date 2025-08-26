import uuid
from pathlib import Path
from typing import Annotated

import modal
import typer

from stopwatch.constants import LLMServerType
from stopwatch.llm_servers import create_dynamic_llm_server_class
from stopwatch.profile import profile
from stopwatch.resources import app, traces_volume

from .utils import config_callback

TRACES_PATH = "/traces"


def llm_server_type_callback(llm_server_type: LLMServerType) -> LLMServerType:
    """Require that llm_server_type is supported for profiling."""

    if llm_server_type in (LLMServerType.vllm, LLMServerType.sglang):
        return llm_server_type.value

    msg = "Profiling is only supported with vLLM or SGLang"
    raise typer.BadParameter(msg)


def profile_cli(
    model: str,
    llm_server_type: Annotated[
        LLMServerType,
        typer.Argument(callback=llm_server_type_callback),
    ],
    *,
    output_path: str = "trace.json.gz",
    gpu: str = "H100",
    server_region: str = "us-chicago-1",
    num_requests: int = 10,
    prompt_tokens: int = 512,
    output_tokens: int = 8,
    llm_server_config: Annotated[
        str | None,
        typer.Option(callback=config_callback),
    ] = None,
) -> None:
    """Run an LLM server alongside the PyTorch profiler."""

    if "env_vars" not in llm_server_config:
        if llm_server_type == LLMServerType.vllm:
            llm_server_config["env_vars"] = {
                "VLLM_TORCH_PROFILER_DIR": TRACES_PATH,
                "VLLM_RPC_TIMEOUT": "1800000",
            }
        elif llm_server_type == LLMServerType.sglang:
            llm_server_config["env_vars"] = {
                "SGLANG_TORCH_PROFILER_DIR": TRACES_PATH,
            }

    name = uuid.uuid4().hex[:4]
    server_cls, _ = create_dynamic_llm_server_class(
        name,
        model,
        gpu=gpu,
        llm_server_type=llm_server_type,
        region=server_region,
        llm_server_config=llm_server_config,
    )

    with modal.enable_output(), app.run():
        fc = profile.spawn(
            endpoint=server_cls().start.get_web_url(),
            model=model,
            num_requests=num_requests,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )

        print(f"Profiler running at {fc.object_id}...")
        trace_path = fc.get()

        with Path(output_path).open("wb") as f:
            for chunk in traces_volume.read_file(trace_path):  # noqa: FURB122
                f.write(chunk)
