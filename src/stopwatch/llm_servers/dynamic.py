import io
import json
from pathlib import Path
from typing import Any

import modal

from stopwatch.constants import HOURS, MINUTES, SECONDS, VersionDefaults
from stopwatch.resources import (
    app,
    db_volume,
    hf_cache_volume,
    hf_secret,
    traces_volume,
    vllm_cache_volume,
)

from .constants import (
    DB_PATH,
    HF_CACHE_PATH,
    SGLANG,
    TENSORRT_LLM,
    TOKASAURUS,
    VLLM,
    VLLM_PD_DISAGGREGATION,
)
from .sglang import SGLangBase, sglang_image_factory
from .tensorrt_llm import TensorRTLLMBase, tensorrt_llm_image_factory
from .tokasaurus import TokasaurusBase, tokasaurus_image_factory
from .vllm import (
    TRACES_PATH,
    VLLM_CACHE_PATH,
    vllm_image_factory,
    vLLMBase,
)
from .vllm_pd_disaggregation import (
    vllm_pd_disaggregation_image_factory,
)


def get_llm_server_class(llm_server_type: str) -> type:
    """Get the base class for creating an LLM server with a given type."""

    llm_server_classes = {
        SGLANG: SGLangBase,
        TENSORRT_LLM: TensorRTLLMBase,
        TOKASAURUS: TokasaurusBase,
        VLLM: vLLMBase,
        VLLM_PD_DISAGGREGATION: vLLMBase,
    }

    return llm_server_classes[llm_server_type]


def get_image(llm_server_type: str, llm_server_config: dict[str, Any]) -> modal.Image:
    """Create an image for an LLM server with a given type and configuration."""
    image_factory_fn = {
        SGLANG: sglang_image_factory,
        TENSORRT_LLM: tensorrt_llm_image_factory,
        TOKASAURUS: tokasaurus_image_factory,
        VLLM: vllm_image_factory,
        VLLM_PD_DISAGGREGATION: vllm_pd_disaggregation_image_factory,
    }

    return image_factory_fn[llm_server_type](
        llm_server_config.get("version", VersionDefaults.LLM_SERVERS[llm_server_type]),
        **llm_server_config.get("image_kwargs", {}),
    )


def get_scaledown_window(llm_server_type: str) -> int:
    """Get the scaledown window for an LLM server with a given type."""

    scaledown_windows = {
        SGLANG: 2 * MINUTES,
        TENSORRT_LLM: 30 * SECONDS,
        TOKASAURUS: 2 * MINUTES,
        VLLM: 30 * SECONDS,
        VLLM_PD_DISAGGREGATION: 30 * SECONDS,
    }

    return scaledown_windows[llm_server_type]


def get_timeout(llm_server_type: str) -> int:
    """Get the timeout for an LLM server with a given type."""

    timeouts = {
        SGLANG: 30 * MINUTES,
        TENSORRT_LLM: 30 * MINUTES,
        TOKASAURUS: 30 * MINUTES,
        VLLM: 1 * HOURS,
        VLLM_PD_DISAGGREGATION: 1 * HOURS,
    }

    return timeouts[llm_server_type]


def get_volumes(llm_server_type: str) -> dict[str, modal.Volume]:
    """Get the volumes for an LLM server with a given type."""

    volumes = {
        DB_PATH: db_volume,
        HF_CACHE_PATH: hf_cache_volume,
    }

    if llm_server_type in (VLLM, VLLM_PD_DISAGGREGATION):
        volumes[VLLM_CACHE_PATH] = vllm_cache_volume

        if llm_server_type == VLLM:
            volumes[TRACES_PATH] = traces_volume

    return volumes


def LLMServerClassFactory(  # noqa: N802
    name: str,
    model: str,
    llm_server_type: str,
    llm_server_config: dict[str, Any] | None = None,
    *,
    parametrized_fn: bool = False,
) -> type:
    """
    Create an LLM server class.

    :param: name: The name of the class.
    :param: model: Name of the model deployed on this server.
    :param: llm_server_type: Type of LLM server (e.g. "vllm", "sglang",
        "tensorrt-llm").
    :param: llm_server_config: Extra configuration for the LLM server.
    :param: parametrized_fn: Set to True to create a parametrized function.
    :return: A server class that hosts an OpenAI-compatible API endpoint.
    """

    server_class = get_llm_server_class(llm_server_type)

    return type(
        name,
        (server_class,),
        {
            "model": model,
            "caller_id": modal.parameter(default="") if parametrized_fn else "",
            "server_config": json.dumps(llm_server_config) or "{}",
            "__annotations__": (
                {
                    "caller_id": str,
                }
                if parametrized_fn
                else {}
            ),
        },
    )


def create_dynamic_llm_server_cls(
    name: str,
    model: str,
    *,
    gpu: str,
    llm_server_type: str,
    cpu: int | None = None,
    memory: int | None = None,
    min_containers: int | None = None,
    max_containers: int | None = None,
    cloud: str | None = None,
    region: str | None = None,
    llm_server_config: dict[str, Any] | None = None,
    max_concurrent_inputs: int = 1000,
    batch: modal.volume.AbstractVolumeUploadContextManager | None = None,
    parametrized_fn: bool = False,
) -> type:
    """
    Create an LLM server class on the fly that will be included in the deployed Modal
    app.
    """

    # Set default values for parameters that are not provided
    num_gpus = 1 if ":" not in gpu else int(gpu.split(":")[1])

    if cpu is None:
        cpu = 4 * num_gpus

    if memory is None:
        memory = 8 * 1024 * num_gpus

    if llm_server_config is None:
        llm_server_config = {}

    # Name must start with "LLM_"
    if not name.startswith("LLM_"):
        name = f"LLM_{name}"

    # Save server config to the DB volume so the class can be recreated later
    server_config = {
        "model": model,
        "llm_server_type": llm_server_type,
        "llm_server_config": llm_server_config,
        "parametrized_fn": parametrized_fn,
    }

    config_buf = io.BytesIO(json.dumps(server_config).encode())
    config_path = f"deployments/{name}.json"

    if batch is None:
        with db_volume.batch_upload(force=True) as b:
            b.put_file(config_buf, config_path)
    else:
        batch.put_file(config_buf, config_path)

    # Deploy the newly created class
    return app.cls(
        image=get_image(llm_server_type, llm_server_config),
        secrets=[hf_secret],
        gpu=gpu,
        volumes=get_volumes(llm_server_type),
        cpu=cpu,
        memory=memory,
        min_containers=min_containers,
        max_containers=max_containers,
        scaledown_window=get_scaledown_window(llm_server_type),
        timeout=get_timeout(llm_server_type),
        cloud=cloud,
        region=region,
    )(
        modal.concurrent(max_inputs=max_concurrent_inputs)(
            LLMServerClassFactory(
                name,
                model,
                llm_server_type,
                llm_server_config,
                parametrized_fn=parametrized_fn,
            ),
        ),
    )


def __getattr__(name: str):  # noqa: ANN202
    """
    When Stopwatch is run, classes will be created dynamically in order to meet the
    needs of the configured benchmark(s). Modal will then need to call these classes
    once the code is deployed. This function allows us to dynamically create these
    classes once Stopwatch has already been deployed.
    """

    if name in globals():
        return globals()[name]

    if name.startswith("LLM_"):
        with Path(DB_PATH).joinpath("deployments", f"{name}.json").open("r") as f:
            server_config = json.load(f)

        return LLMServerClassFactory(
            name,
            server_config["model"],
            server_config["llm_server_type"],
            server_config["llm_server_config"],
            parametrized_fn=server_config["parametrized_fn"],
        )

    msg = f"No attribute {name}"
    raise AttributeError(msg)
