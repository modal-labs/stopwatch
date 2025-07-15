import json
import uuid

import modal

from stopwatch.llm_servers.dynamic import create_dynamic_llm_server_cls
from stopwatch.resources import app


def provision_cli(
    model: str,
    *,
    endpoint_label: str | None = None,
    gpu: str = "H100",
    llm_server_type: str = "vllm",
    cpu: int | None = None,
    memory: int | None = None,
    min_containers: int = 0,
    max_containers: int = 1,
    cloud: str | None = None,
    region: str | None = None,
    llm_server_config: str | None = None,
    max_concurrent_inputs: int = 1000,
) -> None:
    """Deploy an LLM server on Modal."""

    # Pick a random name for the endpoint if not provided
    if endpoint_label is None:
        endpoint_label = uuid.uuid4().hex[:4]

    with modal.enable_output():
        cls = create_dynamic_llm_server_cls(
            endpoint_label,
            model,
            gpu=gpu,
            llm_server_type=llm_server_type,
            cpu=cpu,
            memory=memory,
            min_containers=min_containers,
            max_containers=max_containers,
            cloud=cloud,
            region=region,
            llm_server_config=(
                json.loads(llm_server_config) if llm_server_config else None
            ),
            max_concurrent_inputs=max_concurrent_inputs,
        )

        app.deploy(name="deployment")

        print("Your OpenAI-compatible endpoint is ready at:")
        print(cls().start.get_web_url())
