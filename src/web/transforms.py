"""Functions for transforming columns and rows of benchmark suite data."""

import hashlib
import json
import re
import shlex
from itertools import product
from typing import Any

# In cases where we can't infer the family from the model repo,
# we hard-code the family.
REPO_TO_FAMILY = {
    "RedHatAI/Meta-Llama-3-70B-Instruct-quantized.w4a16": "Llama 3.3",
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
    "deepseek-ai/DeepSeek-V3-0324": "671B-A37B",
    "zed-industries/zeta": "7B",
}


# When inferring the quantization level, we sometimes need to rely
# on a dtype argument. Behavior below is from vLLM's docs as of v0.8.
DTYPE_TO_QUANT = {
    "auto": "f16",
    "bfloat16": "bf16",
    "float": "fp32",
    "float16": "",
    "float32": "fp32",
    "half": "fp16",
}


def transform(df):  # noqa: ANN001, ANN201
    """
    Transform a benchmark suite dataframe into the data model expected by the
    (external) frontend.

    :param: df: A pandas dataframe containing benchmark suite data.
    :return: A pandas dataframe containing the transformed benchmark suite data.
    """

    df = df.rename(
        columns={
            "llm_server_type": "framework",
            "completed_request_rate": "queries_per_second",
        },
    )

    # Parse data configuration into human-readable strings
    df["data"] = df["data"].map(
        lambda x: {k: int(v) for param in x.split(",") for k, v in [param.split("=")]},
    )

    df["task"] = df["data"].map(
        lambda x: (
            "reasoning"
            if x["prompt_tokens"] < x["output_tokens"]
            else "balanced" if x["prompt_tokens"] == x["output_tokens"] else "retrieval"
        ),
    )

    df["total_tokens"] = df["prompt_tokens"] + df["output_tokens"]
    df["tokens"] = (
        df["prompt_tokens"].apply(str).str.cat(df["output_tokens"].apply(str), ";")
    )
    df["generated_tokens"] = df["output_tokens"]
    df["generated_tokens_per_second"] = (
        df["queries_per_second"] * df["generated_tokens"]
    )

    # Parse GPU configuration
    df["gpu_type"] = df["gpu"].map(lambda x: x.split(":")[0].strip("!"))
    df["gpu_count"] = df["gpu"].map(lambda x: int(x.split(":")[1]) if ":" in x else 1)

    # TODO(charles): Hack to remove extra results
    df = df[
        ~((df["gpu"] == "H200:8") & (df["model"] == "deepseek-ai/DeepSeek-V3-0324"))
    ]

    # Extract model properties from repo
    df["model_repo"] = df["model"]
    df["model_family"] = df["model_repo"].map(get_model_family)
    df["model_size"] = df["model_repo"].map(get_model_size)

    # Split LLM server configuration into separate columns
    df["cli_args"] = df["llm_server_config"].map(
        lambda x: " ".join(x["extra_args"]) if "extra_args" in x else None,
    )
    df["env_vars"] = df["llm_server_config"].map(
        lambda x: (
            "\n".join(f"{k}={v}" for k, v in x["env_vars"].items())
            if "env_vars" in x
            else None
        ),
    )
    df["kwargs"] = df["llm_server_config"].map(
        lambda x: json.dumps(x["llm_kwargs"]) if "llm_kwargs" in x else None,
    )
    df["tokenizer"] = df["llm_server_config"].map(lambda x: x.get("tokenizer", None))
    df["framework_version"] = df.apply(
        lambda row: (
            row["version_metadata"].get(row["framework"], None)
            if row["version_metadata"]
            else None
        ),
        axis=1,
    )

    # Extract model quantization
    df["quant"] = df.apply(get_model_quant, axis=1)

    # Create human-readable name for model
    df["model"] = df.apply(get_model_name, axis=1)

    # Add a hash-based id for the experiment (workload + framework + framework config)
    # and for the workload (model + tokens in + out)
    df["id"] = df.apply(hash_config, axis=1)
    workload_config_columns = [
        "model_repo",
        "quant",  # model
        "prompt_tokens",
        "output_tokens",  # data
    ]
    df["workload_id"] = df.apply(
        hash_config,
        axis=1,
        config_columns=workload_config_columns,
    )

    aggregates = ["mean", "p50", "p90", "p95", "p99"]
    metrics_columns = [
        f"{m}_{a}" for m, a in product(["itl", "ttft", "ttlt"], aggregates)
    ]

    for a in aggregates:
        column_name = f"generated_tokens_per_second_per_query_{a}"
        df[column_name] = 1.0 / df[f"itl_{a}"]
        metrics_columns.append(column_name)

    # Add cost-related metrics
    df["gpu_seconds_per_query"] = df["gpu_count"] * 1.0 / df["queries_per_second"]
    df["gpu_seconds_per_1M_total_tokens"] = df["gpu_seconds_per_query"] * (
        1_000_000.0 / df["total_tokens"]
    )

    # Handle non-nullable, non-negative columns -- derived metrics and rates
    strictly_positive_columns = [*metrics_columns, "queries_per_second"]
    for column in strictly_positive_columns:  # all
        df[column] = df[column].apply(lambda x: None if x <= 0 else x)
    df = df.dropna(subset=strictly_positive_columns)

    return df[
        [
            "framework",
            "framework_version",
            "queries_per_second",
            "task",
            "prompt_tokens",
            "output_tokens",
            "generated_tokens",
            "generated_tokens_per_second",
            "total_tokens",
            "tokens",
            "gpu",
            "gpu_type",
            "gpu_count",
            "gpu_seconds_per_query",
            "gpu_seconds_per_1M_total_tokens",
            "model",
            "model_repo",
            "tokenizer",
            "model_family",
            "model_size",
            "quant",
            "cli_args",
            "env_vars",
            "kwargs",
            "rate_type",
            *metrics_columns,
        ]
    ]


def get_model_family(model_repo: str) -> str:
    """
    Get the family of a model from a model repository name.

    :param: model_repo: The name of the model repository.
    :return: The family of the model.
    """

    slug = model_repo.split("/")[-1].lower()
    # happy path
    for model_slug in [
        "llama-3.1",
        "llama-3.2",
        "llama-3.3",
        "gemma-3",
        "ministral",
        "mistral-small-3.1",
    ]:
        if model_slug in slug:
            return " ".join(model_slug.split("-")).title()

    # simple special cases
    if "qwen3" in slug:
        return "Qwen 3"

    if "deepseek-v3" in slug:
        return "DeepSeek-V3"

    # hard special cases, then default to repo
    return REPO_TO_FAMILY.get(model_repo, model_repo.lower())


def get_model_size(model_repo: str) -> str | None:
    """
    Get the size of a model from a model repository name.

    :param: model_repo: The name of the model repository.
    :return: The size of the model.
    """

    model_slug = model_repo.upper().split("/")[-1]
    # match on something that looks like a model size
    size = SIZE_PATTERN.search(model_slug)
    if size:
        return size.group(0).strip("-").upper()
    return REPO_TO_SIZE.get(model_repo)


def get_model_quant(row) -> str | None:  # noqa: ANN001, PLR0911
    """
    Get the quantization level of a model from a row of benchmark suite data.

    :param: row: A row of benchmark suite data, which should contain the LLM server
        type and any server arguments used to run the benchmark.
    :return: The quantization level of the model.
    """

    # try to read from configuration
    dtype = None
    if row["framework"] in ["vllm", "sglang"]:
        cli_args = shlex.split(row.get("cli_args") or "")
        for ii, cli_arg in enumerate(cli_args[:-1]):
            if cli_arg == "--dtype":
                # dtype is needed if we don't find a better indicator
                dtype = cli_args[ii + 1].lower()
            if cli_arg in ("--quantization", "--torchao-config"):
                return cli_args[ii + 1].lower()

    if row["framework"] == "tensorrt-llm":  # noqa: SIM102
        if kwargs := json.loads(row.get("kwargs") or "{}"):  # noqa: SIM102
            if quant_config := kwargs.get("quant_config"):  # noqa: SIM102
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

    if "deepseek" in model_name:
        if "awq" in model_name:
            return "int4"
        return "fp8"

    if dtype:
        return DTYPE_TO_QUANT.get(dtype, dtype)

    # heuristics failed, we don't know
    return None


def get_model_name(row) -> str:  # noqa: ANN001
    """
    Get the name of a model from a row of benchmark suite data.

    :param: row: A row of benchmark suite data, which should contain the model family,
        size, and level of quantization.
    :return: The name of the model.
    """

    return " ".join(
        [row["model_family"], row["model_size"] or "", row["quant"] or ""],
    ).strip()


def hash_config(row: dict[str, Any], config_columns: list[str] | None = None) -> str:
    """
    Generate a unique identifier for a configuration of a benchmark suite data row.

    :param: row: A row of benchmark suite data.
    :param: config_columns: The columns that define the configuration.
    :return: A hash of the configuration.
    """

    # select columns that define the configuration
    if config_columns is None:
        config_columns = [
            "model_repo",
            "quant",  # model
            "framework",  # software
            "prompt_tokens",
            "output_tokens",  # data
            "cli_args",
            "env_vars",
            "kwargs",  # software config
            "gpu",  # hardware
        ]

    values = [row.get(key) for key in config_columns]
    hasher = hashlib.sha256()
    hasher.update(str(values).encode("utf-8"))
    unique_id = hasher.hexdigest()

    return unique_id[:7]
