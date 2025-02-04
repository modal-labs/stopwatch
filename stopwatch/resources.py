import modal

app = modal.App("stopwatch")

hf_secret = modal.Secret.from_name("huggingface-secret")
results_volume = modal.Volume.from_name("stopwatch-results", create_if_missing=True)
tunnel_urls = modal.Dict.from_name("stopwatch-tunnels", create_if_missing=True)
