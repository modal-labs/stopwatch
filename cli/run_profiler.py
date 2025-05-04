from typing import Optional
import json

from stopwatch.resources import app, traces_volume
from stopwatch.run_profiler import run_profiler


@app.local_entrypoint()
def run_profiler_cli(
    model: str,
    gpu: str = "H100",
    server_region: str = "us-chicago-1",
    num_requests: int = 10,
    prompt_tokens: int = 512,
    output_tokens: int = 8,
    llm_server_config: Optional[str] = None,
):
    fc = run_profiler.spawn(
        llm_server_type="vllm",
        model=model,
        gpu=gpu,
        server_region=server_region,
        num_requests=num_requests,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        llm_server_config=json.loads(llm_server_config) if llm_server_config else None,
    )

    print(f"Profiler running at {fc.object_id}...")
    trace_path = fc.get()

    trace = b""
    for chunk in traces_volume.read_file(trace_path):
        trace += chunk

    return trace
