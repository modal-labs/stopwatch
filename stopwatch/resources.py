import modal

app = modal.App("stopwatch")

figures_volume = modal.Volume.from_name("stopwatch-figures", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
results_dict = modal.Dict.from_name("stopwatch-results", create_if_missing=True)
results_volume = modal.Volume.from_name("stopwatch-results", create_if_missing=True)
traces_volume = modal.Volume.from_name("stopwatch-traces", create_if_missing=True)
