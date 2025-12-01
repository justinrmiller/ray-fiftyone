"""
Process a FiftyOne dataset with Ray Data and write results to a new dataset.

This script demonstrates:
1. Reading a FiftyOne dataset using FiftyOneDatasource
2. Processing data with Ray Data (filtering by uniqueness score)
3. Writing results back to FiftyOne using FiftyOneDatasink

Prerequisites:
    pip install fiftyone ray[data] pyarrow

Usage:
    python examples/write_to_fiftyone.py
"""

from __future__ import annotations

from typing import Any

import ray
from ray_fiftyone.data_sink import FiftyOneDatasink
from ray_fiftyone.data_source import FiftyOneDatasource


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
        dataset = fo.load_dataset("quickstart")
        print(f"Dataset has {len(dataset)} samples")


def prepare_for_fiftyone(row: dict[str, Any]) -> dict[str, Any]:
    """
    Prepare a row for writing to FiftyOne by removing system-managed fields.

    FiftyOne automatically manages certain fields (created_at, last_modified_at, etc.)
    and will raise errors if you try to set them manually. This function removes
    those fields before writing.

    In practice, you might also:
    - Run inference to add predictions
    - Compute embeddings
    - Apply custom transformations or enrichments

    Args:
        row: A dictionary row from the Ray Dataset.

    Returns:
        The row with system fields removed.
    """
    system_fields = ["created_at", "last_modified_at", "_id", "id"]
    return {k: v for k, v in row.items() if k not in system_fields}


def main() -> None:
    """Read FiftyOne dataset, process with Ray, write to new FiftyOne dataset."""
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Ensure source dataset exists
    print("Checking for quickstart dataset...")
    ensure_quickstart_dataset()

    # Create datasource to read the quickstart dataset
    print("\n=== Step 1: Reading from FiftyOne ===")
    source = FiftyOneDatasource(
        dataset_name="quickstart",
        # Don't specify fields - read all available fields
        batch_size=100,
    )

    # Read into Ray Dataset
    print("Reading dataset into Ray...")
    ds = ray.data.read_datasource(source)
    print(f"Initial dataset size: {ds.count()} samples")

    # Show schema to see what fields are available
    print("\nDataset schema:")
    print(ds.schema())

    # Process with Ray Data - limit to first 50 samples
    print("\n=== Step 2: Processing with Ray Data ===")
    print("Limiting to first 50 samples...")

    # Get first row to check available fields
    first_row = ds.take(1)[0]

    ds_limited = ds.limit(50)
    limited_count = ds_limited.count()
    print(f"Limited dataset size: {limited_count} samples")

    # Prepare rows for FiftyOne by removing system-managed fields
    ds_processed = ds_limited.map(prepare_for_fiftyone)

    # Write to new FiftyOne dataset
    print("\n=== Step 3: Writing to FiftyOne ===")
    output_dataset_name = "quickstart_filtered"

    # Delete dataset if it already exists to avoid conflicts
    import fiftyone as fo

    if fo.dataset_exists(output_dataset_name):
        print(f"Deleting existing dataset '{output_dataset_name}'...")
        fo.delete_dataset(output_dataset_name)

    # Determine which label fields exist
    label_fields = {}
    if "ground_truth" in first_row:
        label_fields["ground_truth"] = "Detections"
    if "predictions" in first_row:
        label_fields["predictions"] = "Detections"

    sink = FiftyOneDatasink(
        dataset_name=output_dataset_name,
        filepath_field="filepath",
        label_fields=label_fields if label_fields else None,
        overwrite=True,  # Overwrite if dataset already exists
        persistent=True,  # Make dataset persistent
        batch_size=50,
    )

    print(f"Writing to FiftyOne dataset: '{output_dataset_name}'...")
    ds_processed.write_datasink(sink)

    # Verify the write
    print("\n=== Step 4: Verification ===")
    import fiftyone as fo

    output_dataset = fo.load_dataset(output_dataset_name)
    print(
        f"Output dataset '{output_dataset_name}' contains {len(output_dataset)} samples"
    )

    # Show some statistics
    print("\nDataset schema:")
    print(output_dataset.schema)

    print("\nDataset field names:")
    for field_name in output_dataset.get_field_schema().keys():
        print(f"  - {field_name}")

    print("\nSuccess! You can view the dataset in the FiftyOne App:")
    print("  import fiftyone as fo")
    print(f"  dataset = fo.load_dataset('{output_dataset_name}')")
    print("  session = fo.launch_app(dataset)")


if __name__ == "__main__":
    main()
