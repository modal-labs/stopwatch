import modal

app = modal.App()

# Dicts
startup_metrics_dict = modal.Dict.from_name(
    "stopwatch-startup-metrics",
    create_if_missing=True,
)

# Secrets
hf_secret = modal.Secret.from_name("huggingface-secret")

# Volumes
db_volume = modal.Volume.from_name("stopwatch-db", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("stopwatch-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("stopwatch-results", create_if_missing=True)
traces_volume = modal.Volume.from_name("stopwatch-traces", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name(
    "stopwatch-vllm-cache",
    create_if_missing=True,
)
