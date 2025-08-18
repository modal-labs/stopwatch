import uuid

import modal
import pytest

from stopwatch.benchmark import create_dynamic_benchmark_runner_cls
from stopwatch.db import RateType
from stopwatch.llm_servers import create_dynamic_llm_server_cls
from stopwatch.resources import app

DATA = "prompt_tokens=512,output_tokens=128"
DURATION = 10
GPU = "H100!"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

CLIENT_CONFIGS = {
    "sglang": {
        "extra_body": {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Summary",
                    "schema": {
                        "properties": {
                            "summary": {"title": "Summary", "type": "string"},
                        },
                        "required": ["summary"],
                        "title": "Summary",
                        "type": "object",
                    },
                },
            },
        },
    },
    "vllm": {
        "extra_body": {
            "guided_json": {
                "properties": {
                    "summary": {
                        "title": "Summary",
                        "type": "string",
                    },
                },
                "required": ["summary"],
                "title": "Summary",
                "type": "object",
            },
        },
    },
}

SERVER_CONFIGS = {
    "sglang": {
        "extra_args": ["--grammar-backend", "outlines"],
        "image_kwargs": {
            "extra_python_packages": ["outlines"],
            "transformers_version": "4.53.3",
        },
    },
}


@pytest.mark.parametrize("llm_server_type", ["vllm", "sglang"])
def test_structured_outputs(llm_server_type: str) -> None:
    """Test that a quick synchronous benchmark runs successfully."""

    name = uuid.uuid4().hex[:4]
    client_cls = create_dynamic_benchmark_runner_cls(name)
    server_cls = create_dynamic_llm_server_cls(
        name,
        MODEL,
        gpu=GPU,
        llm_server_type=llm_server_type,
        llm_server_config=SERVER_CONFIGS.get(llm_server_type),
    )

    with modal.enable_output(), app.run():
        results = client_cls().run_benchmark.remote(
            endpoint=f"{server_cls().start.get_web_url()}/v1",
            model=MODEL,
            rate_type=RateType.SYNCHRONOUS.value,
            data=DATA,
            duration=DURATION,
            client_config=CLIENT_CONFIGS.get(llm_server_type),
        )

        assert len(results) == 1
        assert results[0]["rate_type"] == RateType.SYNCHRONOUS.value
        assert len(results[0]["results"]) > 0
