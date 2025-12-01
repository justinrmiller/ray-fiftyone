"""
Read FiftyOne quickstart dataset and write to Parquet.

This script demonstrates the basic usage of FiftyOneDatasource to read
a FiftyOne dataset and export it to Parquet format using Ray Data.

Prerequisites:
    pip install fiftyone ray[data] pyarrow

Usage:
    python examples/save_dataset_to_parquet.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import ray
from ray_fiftyone.data_source import FiftyOneDatasource


def flatten_for_parquet(row: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten complex nested structures for Parquet compatibility.

    FiftyOne labels contain nested dicts/structs that may have empty
    child fields, which PyArrow cannot write to Parquet. This function
    converts complex nested objects to JSON strings.

    Args:
        row: A dictionary row from the FiftyOne datasource.

    Returns:
        A flattened dictionary safe for Parquet serialization.
    """
    result: dict[str, Any] = {}

    for key, value in row.items():
        if value is None:
            result[key] = None
        elif isinstance(value, (str, int, float, bool)):
            # Primitive types pass through directly
            result[key] = value
        elif isinstance(value, list):
            # Check if it's a simple list or contains complex objects
            if all(isinstance(x, (str, int, float, bool, type(None))) for x in value):
                result[key] = value
            else:
                # Serialize complex lists to JSON
                result[f"{key}_json"] = json.dumps(value)
        elif isinstance(value, dict):
            # Serialize dicts to JSON strings to avoid empty struct issues
            result[f"{key}_json"] = json.dumps(value)
        else:
            # Fallback: convert to string
            result[key] = str(value)

    return result


def ensure_quickstart_dataset() -> None:
    """Ensure the quickstart dataset is loaded, downloading if necessary."""
    import fiftyone as fo
    import fiftyone.zoo as foz

    # Check if dataset exists and load if not
    if not fo.dataset_exists("quickstart"):
        print("Quickstart dataset not found. Loading from zoo...")
        foz.load_zoo_dataset("quickstart")
    else:
        print("Quickstart dataset already exists.")
        # Load it to ensure it's accessible
        dataset = fo.load_dataset("quickstart")
        print(f"Dataset has {len(dataset)} samples")


def main() -> None:
    """Read FiftyOne quickstart dataset and write to parquet."""
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Ensure dataset exists (this also initializes FiftyOne)
    print("Checking for quickstart dataset...")
    ensure_quickstart_dataset()

    # Create datasource to read the quickstart dataset
    source = FiftyOneDatasource(
        dataset_name="quickstart",
        # Include all fields by leaving fields=None, or specify specific ones:
        # fields=["filepath", "ground_truth", "predictions", "uniqueness"],
        batch_size=100,
    )

    # Read into Ray Dataset
    print("Reading dataset into Ray...")
    ds = ray.data.read_datasource(source)

    print(f"Number of samples: {ds.count()}")

    # Flatten complex structures for Parquet compatibility
    print("Flattening nested structures for Parquet...")
    ds = ds.map(flatten_for_parquet)

    print(f"Flattened schema: {ds.schema()}")

    # Write to parquet
    output_path = "/tmp/quickstart_parquet"

    # Remove existing directory if it exists
    if Path(output_path).exists():
        print(f"Removing existing directory: {output_path}")
        shutil.rmtree(output_path)

    print(f"Writing to parquet at {output_path}...")
    ds.write_parquet(output_path)

    print("Finished writing to parquet.")

    # Verify by reading back
    print("\nVerifying parquet output...")
    ds_verify = ray.data.read_parquet(output_path)
    print(f"Parquet row count: {ds_verify.count()}")
    print("\nFirst row keys:")
    first_row = ds_verify.take(1)[0]
    for key in first_row:
        val = first_row[key]
        val_preview = str(val)[:80] + "..." if len(str(val)) > 80 else str(val)
        print(f"  {key}: {val_preview}")


if __name__ == "__main__":
    main()
