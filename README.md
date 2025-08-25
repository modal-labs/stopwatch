# stopwatch

_A simple solution for benchmarking [vLLM](https://docs.vllm.ai/en/latest/), [SGLang](https://docs.sglang.ai/), and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) on [Modal](https://modal.com/)._ ⏱️

## Setup

### Install dependencies

```bash
pip install -e .
```

## Run a benchmark

To run a single benchmark, you can use the `run_benchmark` command, which will save your results to a local file.
For example, to run a synchronous (one request after another) benchmark with vLLM and save the results to `results.json`:

```bash
LLM_SERVER_TYPE=vllm
MODEL=meta-llama/Llama-3.1-8B-Instruct
OUTPUT_PATH=results.json

stopwatch provision-and-benchmark $MODEL $LLM_SERVER_TYPE --output-path $OUTPUT_PATH
```

Or, to run a fixed-rate (e.g. 5 requests per second) multi-GPU benchmark with SGLang:

```bash
GPU_COUNT=4
GPU_TYPE=H100
LLM_SERVER_TYPE=sglang
RATE_TYPE=constant
REQUESTS_PER_SECOND=5

stopwatch provision-and-benchmark $MODEL $LLM_SERVER_TYPE --output-path $OUTPUT_PATH --gpu "$GPU_TYPE:$GPU_COUNT" --rate-type $RATE_TYPE --rate $REQUESTS_PER_SECOND --llm-server-config "{\"extra_args\": [\"--tp-size\", \"$GPU_COUNT\"]}"
```

Or, to run a throughput (as many requests as the server can handle) test with TensorRT-LLM:

```bash
LLM_SERVER_TYPE=tensorrt-llm
RATE_TYPE=throughput

stopwatch provision-and-benchmark $MODEL $LLM_SERVER_TYPE --output-path $OUTPUT_PATH --rate-type $RATE_TYPE
```

## Run the profiler

To profile a server with the PyTorch profiler, use the following command (only vLLM and SGLang are currently supported):

```bash
LLM_SERVER_TYPE=vllm
MODEL=meta-llama/Llama-3.1-8B-Instruct
NUM_REQUESTS=10
OUTPUT_PATH=trace.json.gz

stopwatch profile $MODEL $LLM_SERVER_TYPE --output-path $OUTPUT_PATH --num-requests $NUM_REQUESTS
```

Once the profiling is done, the trace will be saved to `trace.json.gz`, which you can open and visualize at [https://ui.perfetto.dev](https://ui.perfetto.dev).
Keep in mind that generated traces can get very large, so it is recommended to only send a few requests while profiling.

## Contributing

We welcome contributions, including those that add tuned benchmarks to our collection.
See the [CONTRIBUTING](/CONTRIBUTING.md) file and the [Getting Started](https://github.com/modal-labs/stopwatch/wiki/Getting-Started) document for more details on contributing to Stopwatch.

## License

Stopwatch is available under the MIT license. See the [LICENSE](/LICENSE.md) file for more details.
