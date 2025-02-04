# stopwatch

_A simple solution for benchmarking LLMs on Modal with [guidellm](https://github.com/neuralmagic/guidellm)._ ⏱️

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Deploy to Modal

```bash
modal deploy stopwatch
```

## Run benchmark

```bash
python run_benchmark.py --model meta-llama/Llama-3.1-8B-Instruct -e VLLM_USE_V1=1
```
