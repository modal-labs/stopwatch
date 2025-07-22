import json
import logging
import uuid
from pathlib import Path

import modal

from stopwatch.llm_servers.dynamic import create_dynamic_llm_server_cls
from stopwatch.resources import app, traces_volume
from stopwatch.run_profiler import run_profiler

TRACES_PATH = "/traces"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_profiler_cli(
    model: str,
    *,
    output_path: str = "trace.json.gz",
    detach: bool = False,
    gpu: str = "H100",
    server_region: str = "us-chicago-1",
    num_requests: int = 10,
    prompt_tokens: int = 512,
    output_tokens: int = 8,
    llm_server_type: str = "vllm",
    llm_server_config: str | None = None,
) -> None:
    """Run an LLM server alongside the PyTorch profiler."""

    if llm_server_type != "vllm":
        msg = "Profiling is only supported with vLLM"
        raise ValueError(msg)

    if llm_server_config is not None:
        try:
            llm_server_config = json.loads(llm_server_config)
        except json.JSONDecodeError as e:
            msg = "Invalid JSON for --llm-server-config"
            raise ValueError(msg) from e
    else:
        llm_server_config = {}

    if "env_vars" not in llm_server_config:
        llm_server_config["env_vars"] = {
            "VLLM_TORCH_PROFILER_DIR": TRACES_PATH,
            "VLLM_RPC_TIMEOUT": "1800000",
        }

    name = uuid.uuid4().hex[:4]
    server_cls = create_dynamic_llm_server_cls(
        name,
        model,
        gpu=gpu,
        llm_server_type=llm_server_type,
        region=server_region,
        llm_server_config=llm_server_config,
    )

    with modal.enable_output(), app.run(detach=detach):
        fc = run_profiler.spawn(
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
