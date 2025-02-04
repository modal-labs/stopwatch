import modal

app = modal.App("stopwatch")

results_volume = modal.Volume.from_name("stopwatch-results", create_if_missing=True)
tunnel_urls = modal.Dict.from_name("stopwatch-tunnels", create_if_missing=True)
