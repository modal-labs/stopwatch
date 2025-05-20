"""This Modal script contains tools for extracting, transforming, and loading Stopwatch data.

The final output format is json.
"""

import io
import json
import os
from pathlib import Path
from typing import Optional

import modal

from .resources import db_volume, results_volume, web_app

DB_PATH = "/db"
RESULTS_PATH = "/results"

etl_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("fastapi[standard]", "numpy", "pandas", "SQLAlchemy")
    .add_local_python_source("cli")
)

with etl_image.imports():
    from .db import benchmark_cls_factory, session
    from .transforms import transform

    import pandas as pd


@web_app.function(volumes={RESULTS_PATH: results_volume}, image=etl_image)
def merge_jsonls(jsonl_path: str, json_list: list[dict]) -> list[dict]:
    """Merge JSONL files from the results with an input list of JSON objects."""
    results = [json for json in read_jsonl(Path(RESULTS_PATH) / jsonl_path)]

    return results + json_list


@web_app.local_entrypoint()
def main(
    remote_jsonl_path: str,
    local_jsonl_path: Optional[str] = None,
    to_json: bool = True,
    output_path: str = "query.json",
    key: str = "rows",
    save_locally: bool = False,
):
    """Merge a local and remote jsonl, with optional output locally and to json."""
    import json

    # read any local jsonl
    if local_jsonl_path:
        local_json_list = [json for json in read_jsonl(local_jsonl_path)]
    else:
        local_json_list = []

    # merge local and remote jsonls
    json_list = merge_jsonls.remote(remote_jsonl_path, local_json_list)

    # map to json if requested
    output = (
        json.dumps(json_list_to_json(json_list, outer_key=key))
        if to_json
        else "\n".join(map(json.dumps, json_list))
    )

    # save remotely
    output_path = Path(output_path)
    output_path = (
        output_path.with_suffix(".json")
        if to_json
        else output_path.with_suffix(".jsonl")
    )

    try:
        results_volume.remove_file(str(output_path))
    except Exception:  # file probably doesn't exist
        pass

    with results_volume.batch_upload() as batch:
        batch.put_file(io.BytesIO(output.encode("utf-8")), output_path)

    if save_locally:
        output_path.write_text(output)
        print(f"results saved to {output_path}")
    else:
        print("Retrieve results from")
        print(f"    modal volume get stopwatch-results {output_path}")


def json_list_to_json(jsonls: list[dict], outer_key="rows") -> dict:
    json = {outer_key: jsonls}

    return json


def read_jsonl(file_path: str | Path) -> list[dict]:
    return list(map(json.loads, Path(file_path).read_text().splitlines()))


@web_app.function(
    image=etl_image, volumes={DB_PATH: db_volume, RESULTS_PATH: results_volume}
)
@modal.fastapi_endpoint()
def export_results(suite_ids, verbose: bool = False):
    if not isinstance(suite_ids, list):
        suite_ids = [suite_ids]

    if len(suite_ids) == 0:
        raise ValueError("suite_ids must not be empty")

    # Convert suite_ids to a list of benchmark classes
    if isinstance(suite_ids[0], str):
        suite_ids = [
            benchmark_cls_factory(table_name=suite_id.replace("-", "_") + "_averaged")
            for suite_id in suite_ids
        ]

    # Export all results as a single JSONL blob
    full_buf = io.BytesIO()

    for suite_cls in suite_ids:
        suite_id = suite_cls.__tablename__.rsplit("_averaged")[0].replace("_", "-")
        results = session.query(suite_cls).all()

        if verbose:
            print(f"{len(results)} results to transform from {suite_id}")

        df = pd.DataFrame(
            [{c.name: getattr(r, c.name) for c in r.__table__.columns} for r in results]
        )
        df = transform(df)  # Remap columns, clean up names, etc.

        if verbose:
            print(f"{len(df)} result{'s' if len(df) != 1 else ''} left after cleaning")

        output_path = Path(RESULTS_PATH) / f"{suite_id}.jsonl"

        if verbose:
            print(f"saving to {output_path} on stopwatch-results")

        # Save each individual suite to a separate JSONL file
        suite_buf = io.BytesIO()

        for buf in [full_buf, suite_buf]:
            df.to_json(buf, orient="records", lines=True, force_ascii=False)

        with open(os.path.join(RESULTS_PATH, f"{suite_id}.jsonl"), "w") as f:
            f.write(suite_buf.getvalue().decode("utf-8"))

    return full_buf.getvalue().decode("utf-8")
