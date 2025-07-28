# stopwatch

_A simple solution for benchmarking [vLLM](https://docs.vllm.ai/en/latest/), [SGLang](https://docs.sglang.ai/), and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) on [Modal](https://modal.com/)._ ⏱️

## Setup

### Install dependencies

```bash
pip install -e ".[extras]"
```

## Run a benchmark

To run a single benchmark, you can use the `run_benchmark` command, which will save your results to a local file.
For example, to run a synchronous (one request after another) benchmark with vLLM and save the results to `results.json`:

```bash
LLM_SERVER_TYPE=vllm
MODEL=meta-llama/Llama-3.1-8B-Instruct
OUTPUT_PATH=results.json

stopwatch run-benchmark $MODEL --output-path $OUTPUT_PATH --llm-server-type $LLM_SERVER_TYPE
```

Or, to run a fixed-rate (e.g. 5 requests per second) multi-GPU benchmark with SGLang:

```bash
GPU_COUNT=4
GPU_TYPE=H100
LLM_SERVER_TYPE=sglang
RATE_TYPE=constant
REQUESTS_PER_SECOND=5

stopwatch run-benchmark $MODEL --output-path $OUTPUT_PATH --gpu "$GPU_TYPE:$GPU_COUNT" --model $MODEL --llm-server-type $LLM_SERVER_TYPE --rate-type $RATE_TYPE --rate $REQUESTS_PER_SECOND --llm-server-config "{\"extra_args\": [\"--tp-size\", \"$GPU_COUNT\"]}"
```

Or, to run a throughput (as many requests as the server can handle) test with TensorRT-LLM:

```bash
LLM_SERVER_TYPE=tensorrt-llm
RATE_TYPE=throughput

stopwatch run-benchmark $MODEL --output-path $OUTPUT_PATH --llm-server-type $LLM_SERVER_TYPE --rate-type $RATE_TYPE
```

## Run and plot multiple benchmarks

To run multiple benchmarks at once, first deploy the Datasette UI, which will let you easily view the results later:

```bash
modal deploy -m web
```

Then, start a benchmark suite from a configuration file:

```bash
stopwatch run-benchmark-suite --detach --config-path configs/llama3.yaml
```

Once the suite has finished, you will be given a URL to a UI where you can view your results, and a command to download a JSONL file with your results.

## Run the profiler

To profile vLLM with the PyTorch profiler, use the following command:

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct
NUM_REQUESTS=10
OUTPUT_PATH=trace.json.gz

stopwatch run-profiler $MODEL --output-path $OUTPUT_PATH --num-requests $NUM_REQUESTS
```

Once the profiling is done, the trace will be saved to `trace.json.gz`, which you can open and visualize at [https://ui.perfetto.dev](https://ui.perfetto.dev).
Keep in mind that generated traces can get very large, so it is recommended to only send a few requests while profiling.

## Contributing

We welcome contributions, including those that add tuned benchmarks to our collection.
See the [CONTRIBUTING](/CONTRIBUTING.md) file and the [Getting Started](https://github.com/modal-labs/stopwatch/wiki/Getting-Started) document for more details on contributing to Stopwatch.

## License

Stopwatch is available under the MIT license. See the [LICENSE](/LICENSE.md) file for more details.
