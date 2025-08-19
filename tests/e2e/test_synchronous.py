import pytest

from stopwatch.cli import run_benchmark_cli
from stopwatch.constants import LLMServerType, RateType


@pytest.mark.parametrize(
    "llm_server_type",
    [
        LLMServerType.vllm,
        LLMServerType.sglang,
        LLMServerType.tensorrt_llm,
        LLMServerType.tokasaurus,
    ],
)
def test_llama(llm_server_type: LLMServerType) -> None:
    """Test that a quick synchronous benchmark runs successfully."""

    results = run_benchmark_cli(
        "meta-llama/Llama-3.1-8B-Instruct",
        llm_server_type,
        gpu="H100!",
        duration=10,
        server_cloud="oci",
        client_config=(
            {
                "remove_from_body": ["max_completion_tokens", "stream"],
            }
            if llm_server_type == LLMServerType.tokasaurus
            else None
        ),
    )

    assert len(results) == 1
    assert results[0]["rate_type"] == RateType.synchronous.value
    assert len(results[0]["results"]) > 0
