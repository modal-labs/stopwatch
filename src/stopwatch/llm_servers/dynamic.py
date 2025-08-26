import io
import json
from pathlib import Path
from typing import Any

import modal

from stopwatch.constants import (
    DB_PATH,
    HF_CACHE_PATH,
    HOURS,
    MINUTES,
    SECONDS,
    TRACES_PATH,
    VLLM_CACHE_PATH,
    LLMServerType,
)
from stopwatch.resources import (
    app,
    db_volume,
    hf_cache_volume,
    hf_secret,
    traces_volume,
    vllm_cache_volume,
)

from .sglang import SGLangBase, sglang_image_factory
from .tensorrt_llm import TensorRTLLMBase, tensorrt_llm_image_factory
from .tokasaurus import TokasaurusBase, tokasaurus_image_factory
from .vllm import (
    vllm_image_factory,
    vLLMBase,
)
from .vllm_pd_disaggregation import (
    vllm_pd_disaggregation_image_factory,
)


def get_llm_server_class(llm_server_type: LLMServerType) -> type:
    """Get the base class for creating an LLM server with a given type."""

    llm_server_classes = {
        LLMServerType.sglang: SGLangBase,
        LLMServerType.tensorrt_llm: TensorRTLLMBase,
        LLMServerType.tokasaurus: TokasaurusBase,
        LLMServerType.vllm: vLLMBase,
        LLMServerType.vllm_pd_disaggregation: vLLMBase,
    }

    return llm_server_classes[llm_server_type]


def get_image(
    llm_server_type: LLMServerType,
    llm_server_config: dict[str, Any],
) -> modal.Image:
    """Create an image for an LLM server with a given type and configuration."""
    image_factory_fn = {
        LLMServerType.sglang: sglang_image_factory,
        LLMServerType.tensorrt_llm: tensorrt_llm_image_factory,
        LLMServerType.tokasaurus: tokasaurus_image_factory,
        LLMServerType.vllm: vllm_image_factory,
        LLMServerType.vllm_pd_disaggregation: vllm_pd_disaggregation_image_factory,
    }

    return image_factory_fn[llm_server_type](
        llm_server_config.get("version", llm_server_type.get_version()),
        **llm_server_config.get("image_kwargs", {}),
    )


def get_scaledown_window(llm_server_type: LLMServerType) -> int:
    """Get the scaledown window for an LLM server with a given type."""

    scaledown_windows = {
        LLMServerType.sglang: 2 * MINUTES,
        LLMServerType.tensorrt_llm: 30 * SECONDS,
        LLMServerType.tokasaurus: 2 * MINUTES,
        LLMServerType.vllm: 30 * SECONDS,
        LLMServerType.vllm_pd_disaggregation: 30 * SECONDS,
    }

    return scaledown_windows[llm_server_type]


def get_timeout(llm_server_type: LLMServerType) -> int:
    """Get the timeout for an LLM server with a given type."""

    timeouts = {
        LLMServerType.sglang: 30 * MINUTES,
        LLMServerType.tensorrt_llm: 30 * MINUTES,
        LLMServerType.tokasaurus: 30 * MINUTES,
        LLMServerType.vllm: 1 * HOURS,
        LLMServerType.vllm_pd_disaggregation: 1 * HOURS,
    }

    return timeouts[llm_server_type]


def get_volumes(llm_server_type: LLMServerType) -> dict[str, modal.Volume]:
    """Get the volumes for an LLM server with a given type."""

    volumes = {
        DB_PATH: db_volume,
        HF_CACHE_PATH: hf_cache_volume,
    }

    if llm_server_type in (LLMServerType.vllm, LLMServerType.vllm_pd_disaggregation):
        volumes[VLLM_CACHE_PATH] = vllm_cache_volume

    if llm_server_type in (LLMServerType.vllm, LLMServerType.sglang):
        volumes[TRACES_PATH] = traces_volume

    return volumes


def LLMServerClassFactory(  # noqa: N802
    name: str,
    model: str,
    llm_server_type: LLMServerType,
    llm_server_config: dict[str, Any] | None = None,
) -> type:
    """
    Create an LLM server class.

    :param: name: The name of the class.
    :param: model: Name of the model deployed on this server.
    :param: llm_server_type: Type of LLM server.
    :param: llm_server_config: Extra configuration for the LLM server.
    :return: A server class that hosts an OpenAI-compatible API endpoint.
    """

    server_class = get_llm_server_class(llm_server_type)

    return type(
        name,
        (server_class,),
        {
            "model": model,
            "server_config": json.dumps(llm_server_config) or "{}",
            "server_id": name,
            "__annotations__": {},
        },
    )


def create_dynamic_llm_server_class(
    name: str,
    model: str,
    *,
    gpu: str,
    llm_server_type: LLMServerType,
    cpu: int | None = None,
    memory: int | None = None,
    min_containers: int | None = None,
    max_containers: int | None = None,
    cloud: str | None = None,
    region: str | None = None,
    llm_server_config: dict[str, Any] | None = None,
    max_concurrent_inputs: int = 1000,
    batch: modal.volume.AbstractVolumeUploadContextManager | None = None,
) -> tuple[type, str]:
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
        "llm_server_type": llm_server_type.value,
        "llm_server_config": llm_server_config,
    }

    config_buf = io.BytesIO(json.dumps(server_config).encode())
    config_path = f"deployments/{name}.json"

    if batch is None:
        with db_volume.batch_upload(force=True) as b:
            b.put_file(config_buf, config_path)
    else:
        batch.put_file(config_buf, config_path)

    # Deploy the newly created class
    return (
        app.cls(
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
                ),
            ),
        ),
        name,
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
            LLMServerType(server_config["llm_server_type"]),
            server_config["llm_server_config"],
        )

    msg = f"No attribute {name}"
    raise AttributeError(msg)
