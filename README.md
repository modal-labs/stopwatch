# stopwatch

_A simple solution for benchmarking [vLLM](https://docs.vllm.ai/en/latest/) and [trtLLM](https://github.com/NVIDIA/TensorRT-LLM) on [Modal](https://modal.com/) with [guidellm](https://github.com/neuralmagic/guidellm)._ ⏱️

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Deploy to Modal

```bash
modal deploy -m stopwatch
```

## Run benchmark

To run a single benchmark, you can use the `run-benchmark` command.
For example:

```bash
$ python cli.py run-benchmark --model meta-llama/Llama-3.1-8B-Instruct -e VLLM_USE_V1=1
Benchmark running at fc-XXXXXXXX
```

You can then download the results of the benchmark from the `stopwatch-results` volume:

```bash
modal volume get stopwatch-results fc-XXXXXXXX.json
```

## Run and plot multiple benchmarks

To run multiple benchmarks at once, you can use the `run-benchmark-suite` command, along with a configuration file. For example, running the following command...

```bash
python cli.py run-benchmark-suite configs/vllm-v1-engine.yaml
```
## Run profiler

To profile vLLM with the PyTorch profiler, use the following command:

```bash
python cli.py run-profiler --model meta-llama/Llama-3.1-8B-Instruct --num-requests 10
```

Once the profiling is done, you will be prompted to download the generated trace and reveal it in Finder.
Keep in mind that generated traces can get very large, so it is recommended to only send a few requests while profiling.
Traces can then be visualized at [https://ui.perfetto.dev](https://ui.perfetto.dev).

## License

Stopwatch is available under the MIT license. See the [LICENSE](/LICENSE.md) file for more details.
