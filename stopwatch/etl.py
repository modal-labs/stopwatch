"""This Modal script contains tools for extracting, transforming, and loading Stopwatch data.

The final output format is json.
"""

import os
import io
from pathlib import Path
from typing import Optional

import modal

from .db import benchmark_cls_factory, session
from .resources import app, db_volume, results_volume
from .transforms import transform

DB_PATH = "/db"
RESULTS_PATH = "/results"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("numpy", "pandas", "SQLAlchemy")
    .add_local_python_source("cli")
)


@app.function(volumes={RESULTS_PATH: results_volume}, image=image)
def merge_jsonls(jsonl_path: str, json_list: list[dict]) -> list[dict]:
    """Merge JSONL files from the results with an input list of JSON objects."""
    results = [json for json in read_jsonl(Path(RESULTS_PATH) / jsonl_path)]

    return results + json_list


@app.local_entrypoint()
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
    import json

    return list(map(json.loads, Path(file_path).read_text().splitlines()))


@app.function(image=image, volumes={RESULTS_PATH: results_volume, DB_PATH: db_volume})
def extract_transform_suite_table(suite_id_or_cls, verbose: bool = False):
    import pandas as pd

    if isinstance(suite_id_or_cls, str):  # id
        suite_id = suite_id_or_cls
        SuiteAveragedBenchmark = benchmark_cls_factory(
            table_name=suite_id.replace("-", "_") + "_averaged"
        )
    else:  # cls
        SuiteAveragedBenchmark = suite_id_or_cls
        suite_id = SuiteAveragedBenchmark.__tablename__.rsplit("_averaged")[0].replace(
            "_", "-"
        )
    results = session.query(SuiteAveragedBenchmark).all()

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

    # Save selected columns to JSONL file
    df.to_json(
        os.path.join(RESULTS_PATH, f"{suite_id}.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False,
    )
    results_volume.commit()
