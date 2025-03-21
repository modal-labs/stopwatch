import asyncio

from modal._utils.async_utils import TaskContext
from modal.functions import synchronize_api
import modal


async def _as_completed(function_calls):
    async with TaskContext() as tc:
        tasks = [(fc.object_id, tc.create_task(fc.get())) for fc in function_calls]

        async for finished_task in asyncio.as_completed([task for _, task in tasks]):
            try:
                result = await finished_task
            except (
                modal.exception.FunctionTimeoutError,
                modal.exception.RemoteError,
            ) as exc:
                result = exc

            for function_call_id, task in tasks:
                if finished_task is task:
                    yield function_call_id, result


as_completed = synchronize_api(_as_completed)
