import pytest

from stopwatch.cli import provision_and_benchmark_cli
from stopwatch.constants import LLMServerType

CLIENT_CONFIGS = {
    LLMServerType.sglang: {
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
    LLMServerType.vllm: {
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
    LLMServerType.sglang: {
        "extra_args": ["--grammar-backend", "outlines"],
        "image_kwargs": {
            "extra_python_packages": ["outlines", "transformers==4.53.3"],
        },
    },
}


@pytest.mark.parametrize("llm_server_type", [LLMServerType.vllm, LLMServerType.sglang])
def test_structured_outputs(llm_server_type: LLMServerType) -> None:
    """Test that a quick synchronous benchmark runs successfully."""

    results = provision_and_benchmark_cli(
        "meta-llama/Llama-3.1-8B-Instruct",
        llm_server_type,
        gpu="H100!",
        duration=10,
        llm_server_config=SERVER_CONFIGS.get(llm_server_type),
        client_config=CLIENT_CONFIGS.get(llm_server_type),
    )

    # Only one benchmark should have been run
    assert len(results) == 1

    # At least one successful request should have been made
    assert results[0]["run_stats"]["requests_made"]["successful"] > 0
