# Credit: https://github.com/vllm-project/vllm/blob/559756214b770d0405939a05172804221c2f5677/benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
from quart import Quart, Response, make_response, request

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)
logger = logging.getLogger(__name__)


async def forward_request(
    url: str,
    *,
    body: dict[str, Any] | None = None,
    method: str = "POST",
) -> AsyncGenerator[bytes, None]:
    """Forward a request to the prefill and decode servers."""

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        async with session.request(
            method=method,
            url=url,
            json=body,
            headers=headers,
        ) as response:
            if response.status == 200:  # noqa: PLR2004
                if response.headers.get("Transfer-Encoding") == "chunked":
                    # if True:
                    async for chunk_bytes in response.content.iter_chunked(1024):
                        yield chunk_bytes
                else:
                    content = await response.read()
                    yield content


@app.route("/v1/completions", methods=["POST"])
async def handle_request() -> Response:
    """
    Forward the request to the prefill and decode servers, and then return the response
    from the decode server.
    """

    try:
        original_request_data = await request.get_json()

        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request["max_tokens"] = 1

        # finish prefill
        async for _ in forward_request(
            "http://localhost:8100/v1/completions",
            body=prefill_request,
        ):
            continue

        # return from decode server
        generator = forward_request(
            "http://localhost:8200/v1/completions",
            body=original_request_data,
        )
        response = await make_response(generator)
        response.timeout = None
    except Exception:
        logger.exception("Error occurred in disagg prefill proxy server")
    else:
        return response


@app.route("/v1/models")
async def handle_models() -> Response:
    """Return the models from the prefill server."""
    generator = forward_request("http://localhost:8100/v1/models", method="GET")
    response = await make_response(generator)
    response.timeout = None
    return response


@app.route("/ping")
def handle_ping() -> str:
    """Return a 200 status code."""
    return "pong"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
