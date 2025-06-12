import modal

app = modal.App()
web_app = modal.App("stopwatch")

db_volume = modal.Volume.from_name("stopwatch-db", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("stopwatch-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
results_volume = modal.Volume.from_name("stopwatch-results", create_if_missing=True)
traces_volume = modal.Volume.from_name("stopwatch-traces", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name(
    "stopwatch-vllm-cache", create_if_missing=True,
)
