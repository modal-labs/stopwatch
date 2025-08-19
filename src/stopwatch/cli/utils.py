import json

import typer


def config_callback(config: str | None) -> dict:
    """Parse JSON config strings into dicts."""

    if isinstance(config, dict):
        return config

    if config is None:
        return {}

    try:
        return json.loads(config)
    except json.JSONDecodeError as err:
        msg = "Must be a valid JSON string"
        raise typer.BadParameter(msg) from err
