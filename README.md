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
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
OUTPUT_PATH=results.json

modal run -w $OUTPUT_PATH -m cli.run_benchmark --model $MODEL --llm-server-type vllm
```

Or, to run a fixed-rate (e.g. 5 requests per second) multi-GPU benchmark with SGLang:

```bash
GPU_COUNT=4
MODEL=meta-llama/Llama-3.3-70B-Instruct
REQUESTS_PER_SECOND=5

modal run -w $OUTPUT_PATH -m cli.run_benchmark --gpu "H100:$GPU_COUNT" --model $MODEL --llm-server-type sglang --rate-type constant --rate $REQUESTS_PER_SECOND --llm-server-config "{\"extra_args\": [\"--tp-size\", \"$GPU_COUNT\"]}"
```

Or, to run a throughput (as many requests as the server can handle) test with TensorRT-LLM:

```bash
modal run -w $OUTPUT_PATH -m cli.run_benchmark --model $MODEL --llm-server-type tensorrt-llm --rate-type throughput
```

## Run and plot multiple benchmarks

To run multiple benchmarks at once, first deploy the Datasette UI, which will let you easily view the results later:

```bash
modal deploy -m stopwatch
```

Then, start a benchmark suite from a configuration file:

```bash
modal run -d -m cli.run_benchmark_suite --config-path configs/llama3.yaml
```

Once the suite has finished, you will be given a URL to a UI where you can view your results, and a command to download a JSONL file with your results.

## Run the profiler

To profile vLLM with the PyTorch profiler, use the following command:

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct
OUTPUT_PATH=trace.json.gz

modal run -w $OUTPUT_PATH -m cli.run_profiler --model $MODEL --num-requests 10
```

Once the profiling is done, the trace will be saved to `trace.json.gz`, which you can open and visualize at [https://ui.perfetto.dev](https://ui.perfetto.dev).
Keep in mind that generated traces can get very large, so it is recommended to only send a few requests while profiling.

## Contributing

We welcome contributions, including those that add tuned benchmarks to our collection.
See the [CONTRIBUTING](/CONTRIBUTING.md) file and the [Getting Started](https://github.com/modal-labs/stopwatch/wiki/Getting-Started) document for more details on contributing to Stopwatch.

## License

Stopwatch is available under the MIT license. See the [LICENSE](/LICENSE.md) file for more details.
