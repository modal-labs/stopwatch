"""
Modal script containing tools for extracting, transforming, and loading Stopwatch data.

The final output format is JSONL.
"""

import contextlib
import io
import json
import logging
from pathlib import Path

import modal

from stopwatch.resources import db_volume, results_volume

from .resources import web_app

DB_PATH = "/db"
RESULTS_PATH = "/results"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

etl_image = modal.Image.debian_slim(python_version="3.13").uv_pip_install(
    "fastapi[standard]",
    "numpy",
    "pandas",
    "SQLAlchemy",
)

with etl_image.imports():
    from stopwatch.db import benchmark_class_factory, session

    from .transforms import transform


@web_app.function(volumes={RESULTS_PATH: results_volume}, image=etl_image)
def merge_jsonls(jsonl_path: str, json_list: list[dict]) -> list[dict]:
    """
    Merge JSONL files from the results with an input list of JSON objects.

    :param: jsonl_path: The path to the JSONL file.
    :param: json_list: A list of JSON objects.
    :return: A list of JSON objects.
    """

    results = read_jsonl(Path(RESULTS_PATH) / jsonl_path)
    return results + json_list


@web_app.local_entrypoint()
def main(
    remote_jsonl_path: str,
    local_jsonl_path: str | None = None,
    *,
    to_json: bool = True,
    output_path: str = "query.json",
    key: str = "rows",
    save_locally: bool = False,
) -> None:
    """
    Merge a local and remote JSONL, with optional local output and to JSON.

    :param: remote_jsonl_path: The path to the remote JSONL file.
    :param: local_jsonl_path: The path to the local JSONL file.
    :param: to_json: Whether to convert the output to JSON.
    :param: output_path: The path to the output file.
    :param: key: The key to use for the outer JSON object.
    :param: save_locally: Whether to save the output locally.
    """

    import json

    # read any local jsonl
    local_json_list = read_jsonl(local_jsonl_path) if local_jsonl_path else []

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

    with contextlib.suppress(Exception):  # file probably doesn't exist
        results_volume.remove_file(str(output_path))

    with results_volume.batch_upload() as batch:
        batch.put_file(io.BytesIO(output.encode("utf-8")), output_path)

    if save_locally:
        output_path.write_text(output)
        logger.info("Results saved to %s", output_path)
    else:
        logger.info("Retrieve results from")
        logger.info("    modal volume get stopwatch-results %s", output_path)


def json_list_to_json(jsonls: list[dict], outer_key: str = "rows") -> dict:
    """
    Convert a list of dictionaries to a JSON object with a single key.

    :param: jsonls: A list of dictionaries.
    :param: outer_key: The key to use for the outer JSON object.
    :return: A JSON object with a single key.
    """

    return {outer_key: jsonls}


def read_jsonl(file_path: str | Path) -> list[dict]:
    """
    Read a JSONL file and return a list of dictionaries.

    :param: file_path: The path to the JSONL file.
    :return: A list of dictionaries.
    """

    return list(map(json.loads, Path(file_path).read_text().splitlines()))


@web_app.function(
    image=etl_image,
    volumes={DB_PATH: db_volume, RESULTS_PATH: results_volume},
)
@modal.fastapi_endpoint()
def export_results(suite_ids: str, *, verbose: bool = False):  # noqa: ANN201
    """
    Export results from the database as a JSONL blob.

    :param: suite_ids: A list of suite IDs or benchmark classes.
    :param: verbose: Whether to log verbose output.
    :return: A JSONL blob.
    """

    import pandas as pd
    from fastapi import Response

    if isinstance(suite_ids, str):
        suite_ids = suite_ids.split(",")
    elif not isinstance(suite_ids, list):
        suite_ids = [suite_ids]

    if len(suite_ids) == 0:
        msg = "suite_ids must not be empty"
        raise ValueError(msg)

    # Convert suite_ids to a list of benchmark classes
    if isinstance(suite_ids[0], str):
        suite_ids = [
            benchmark_class_factory(table_name=suite_id.replace("-", "_") + "_averaged")
            for suite_id in suite_ids
        ]

    # Export all results as a single JSONL blob
    full_buf = io.BytesIO()

    for suite_cls in suite_ids:
        suite_id = suite_cls.__tablename__.rsplit("_averaged")[0].replace("_", "-")
        results = session.query(suite_cls).all()

        if verbose:
            logger.info("%d results to transform from %s", len(results), suite_id)

        df = pd.DataFrame(
            [
                {c.name: getattr(r, c.name) for c in r.__table__.columns}
                for r in results
            ],
        )
        df = transform(df)  # Remap columns, clean up names, etc.

        if verbose:
            logger.info(
                "%d result%s left after cleaning",
                len(df),
                "s" if len(df) != 1 else "",
            )

        output_path = Path(RESULTS_PATH) / f"{suite_id}.jsonl"

        if verbose:
            logger.info("Saving to %s on stopwatch-results", output_path)

        # Save each individual suite to a separate JSONL file
        suite_buf = io.BytesIO()

        for buf in [full_buf, suite_buf]:
            df.to_json(buf, orient="records", lines=True, force_ascii=False)

        with (Path(RESULTS_PATH) / f"{suite_id}.jsonl").open("w") as f:
            f.write(suite_buf.getvalue().decode("utf-8"))

    return Response(
        content=full_buf.getvalue().decode("utf-8"),
        media_type="application/jsonl",
    )
