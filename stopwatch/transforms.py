"""Functions for transforming columns and rows of benchmark suite data."""

from itertools import product
import json
import re
import shlex

# Define the model "family", mostly branding so manual
REPO_TO_FAMILY = {
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16": "Llama 3.1",
    "RedHatAI/Meta-Llama-3-70B-Instruct-quantized.w4a16": "Llama 3.3",
    "Qwen/Qwen3-0.6B-FP8": "Qwen 3",
    "Qwen/Qwen3-235B-A22B": "Qwen 3",
    "cognitivecomputations/DeepSeek-V3-0324-AWQ": "DeepSeek-V3",
    "google/gemma-3-4b-it": "Gemma 3",
    "google/gemma-3-12b-it": "Gemma 3",
    "google/gemma-3-27b-it": "Gemma 3",
    "hugging-quants/meta-llama-3.1-8B-Instruct-awq-int4": "Llama 3.1",
    "mistralai/Ministral-8B-Instruct-2410": "Ministral",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "Mistral Small 3.1",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1",
    "meta-llama/Llama-3.3-70B-instruct": "Llama 3.3",
    "zed-industries/zeta": "Qwen 2.5",
}

# We attempt to infer the size from the model repo by looking for
# <number><unit>[-A?<number><unit>]
SIZE_PATTERN = re.compile(
    r"-"  # leading dash
    r"\d+(?:\.\d+)?"  # integer or decimal number
    r"[MB]"  # unit: M or B
    r"(?:"  # optional second “-A?number+unit”
    r"-"  #   dash
    r"(?:A)?"  #   optional “A”
    r"\d+(?:\.\d+)?"  #   integer or decimal
    r"[MB]"  #   unit
    r")?",  # end optional group
    flags=re.IGNORECASE,
)


# In cases where we can't infer the size from the model repo,
# we hard-code a value.
REPO_TO_SIZE = {
    "cognitivecomputations/DeepSeek-V3-0324-AWQ": "671B-A37B",
    "zed-industries/zeta": "7B",
}


# When inferring the quantization level, we sometimes need to rely
# on a dtype argument. Behavior below is from vLLM's docs as of v0.8.
DTYPE_TO_QUANT = {
    "auto": "f16",
    "bfloat16": "fp16",
    "float": "fp32",
    "float16": "",
    "float32": "fp32",
    "half": "fp16",
}


def transform(df):
    """Transform a benchmark suite dataframe into the data model expected by the (external) frontend."""
    df = df.rename(
        columns={
            "llm_server_type": "framework",
            "completed_request_rate": "queries_per_second",
        }
    )

    # Parse data configuration into human-readable strings
    df["data"] = df["data"].map(
        lambda x: {k: int(v) for param in x.split(",") for k, v in [param.split("=")]}
    )

    df["task"] = df["data"].map(
        lambda x: (
            "reasoning"
            if x["prompt_tokens"] < x["output_tokens"]
            else "balanced"
            if x["prompt_tokens"] == x["output_tokens"]
            else "retrieval"
        )
    )

    df["total_tokens"] = df["prompt_tokens"] + df["output_tokens"]
    df["generated_tokens"] = df["output_tokens"]

    # Parse GPU configuration
    df["gpu_type"] = df["gpu"].map(lambda x: x.split(":")[0].strip("!"))
    df["gpu_count"] = df["gpu"].map(lambda x: int(x.split(":")[1]) if ":" in x else 1)

    # Extract model properties from repo
    df["model_repo"] = df["model"]
    df["model_family"] = df["model_repo"].map(get_model_family)
    df["model_size"] = df["model_repo"].map(get_model_size)

    # Split LLM server configuration into separate columns
    df["cli_args"] = df["llm_server_config"].map(
        lambda x: " ".join(x["extra_args"]) if "extra_args" in x else None
    )
    df["env_vars"] = df["llm_server_config"].map(
        lambda x: (
            "\n".join(f"{k}={v}" for k, v in x["env_vars"].items())
            if "env_vars" in x
            else None
        )
    )
    df["kwargs"] = df["llm_server_config"].map(
        lambda x: json.dumps(x["llm_kwargs"]) if "llm_kwargs" in x else None
    )

    # Extract model quantization
    df["quant"] = df.apply(get_model_quant, axis=1)

    # Create human-readable name for model
    df["model"] = df.apply(get_model_name, axis=1)

    return df[
        [
            *[
                f"{m}_{a}"
                for m, a in product(
                    ["itl", "ttft", "ttlt"], ["mean", "p50", "p90", "p95", "p99"]
                )
            ],
            "framework",
            "queries_per_second",
            "task",
            "prompt_tokens",
            "output_tokens",
            "generated_tokens",
            "total_tokens",
            "gpu",
            "gpu_type",
            "gpu_count",
            "model",
            "model_repo",
            "model_family",
            "model_size",
            "quant",
            "cli_args",
            "env_vars",
            "kwargs",
            "rate_type",
        ]
    ]


def get_model_family(model_repo):
    return REPO_TO_FAMILY.get(model_repo, model_repo.lower())


def get_model_size(model_repo):
    model_slug = model_repo.upper().split("/")[-1]
    # match on something that looks like a model size
    size = SIZE_PATTERN.search(model_slug)
    if size:
        return size.group(0).strip("-").upper()
    else:
        return REPO_TO_SIZE.get(model_repo, None)


def get_model_quant(row):
    # try to read from configuration
    dtype = None
    if row["framework"] in ["vllm", "sglang"]:
        cli_args = shlex.split(row.get("cli_args") or "")
        for ii, cli_arg in enumerate(cli_args[:-1]):
            if cli_arg == "--dtype":
                # dtype is needed if we don't find a better indicator
                dtype = cli_args[ii + 1].lower()
            if (cli_arg == "--quantization") or (cli_arg == "--torchao-config"):
                return cli_args[ii + 1].lower()

    if row["framework"] == "tensorrt-llm":
        if kwargs := row.get("llm_kwargs") or {}:
            if quant_config := kwargs.get("quant_config"):
                if quant_algo := quant_config.get("quant_algo"):
                    return quant_algo.lower()

    # try to infer from model name
    model_name = (row.get("model_repo") or "").lower().split("/")[-1]
    for quant in ["int4", "fp4", "fp6", "int8", "fp8", "bf16", "fp16", "fp32"]:
        if quant in model_name:
            return quant

    llm_compressor_to_quant = {"w4a16": "int4", "w8a8": "fp8"}
    for llm_compressor_name, quant in llm_compressor_to_quant.items():
        if llm_compressor_name in model_name:
            return quant

    if "gguf" in model_name:
        # look for q4_0, q6_K_M, etc
        matches = re.findall(r"q(\d)_", model_name)
        if matches:
            return f"int{matches[-1]}"  # digit

    if "awq" in model_name:
        if "deepseek" in model_name:
            return "int4"

    if dtype:
        return DTYPE_TO_QUANT.get(dtype, dtype)
    else:  # heuristics failed, we don't know
        return None


def get_model_name(row):
    return " ".join(
        [row["model_family"], row["model_size"] or "", row["quant"] or ""]
    ).strip()
