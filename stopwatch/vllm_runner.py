import logging
import subprocess

import modal

from .resources import app, tunnel_urls

CONTAINER_IDLE_TIMEOUT = 30  # 30 seconds
TIMEOUT = 60 * 60  # 1 hour
VLLM_PORT = 8000


vllm_image = modal.Image.from_registry(
    "vllm/vllm-openai",
    setup_dockerfile_commands=[
        "RUN ln -s /usr/bin/python3 /usr/bin/python3.vllm",
    ],
    add_python="3.12",
).dockerfile_commands("ENTRYPOINT []")


@app.cls(
    gpu=modal.gpu.H100(),
    image=vllm_image,
    container_idle_timeout=CONTAINER_IDLE_TIMEOUT,
    timeout=TIMEOUT,
)
class vLLM:
    """
    A Modal class that runs a vLLM server. The endpoint is exposed via a
    tunnel, the URL for which is stored in a shared dict.
    """

    @modal.method()
    def start(self, model: str, caller_id: str):
        """Start the vLLM server.

        Args:
            model (str): Name of the model to run.
            caller_id (str): ID of the function call that started the vLLM server.
        """

        self.caller_id = caller_id

        with modal.forward(VLLM_PORT) as tunnel:
            logging.info(f"Starting vLLM server at {tunnel.url}")

            # Save tunnel URL so that the benchmarking runner can access it
            tunnel_urls[caller_id] = f"{tunnel.url}/v1"

            # Start vLLM server
            subprocess.run(
                [
                    "python3.vllm",
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    model,
                ]
            )

    @modal.exit()
    def shutdown(self):
        # Remove tunnel URL from dict
        if hasattr(self, "caller_id"):
            tunnel_urls.pop(self.caller_id)

        # Kill vLLM server
        subprocess.run(["pkill", "-9", "python3.vllm"])
