"""
Ray Data sink for FiftyOne (Voxel51) datasets.

This module provides integration between Ray Data and FiftyOne, enabling
distributed processing of computer vision datasets using Ray's parallel
execution capabilities.

Usage:
    # Writing to FiftyOne
    ds.write_datasink(
        FiftyOneDatasink(
            dataset_name="my_output_dataset",
            filepath_field="filepath",
        )
    )
"""

from __future__ import annotations

import logging
import ray_fiftyone.helpers as helpers
from typing import (
    TYPE_CHECKING,
    Any,
)
from collections.abc import Iterable

import ray

import pyarrow as pa

from ray.data.block import Block
from ray.data.datasource import Datasink

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import TaskContext

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# FiftyOne Datasink (Writer)
# -----------------------------------------------------------------------------


class FiftyOneDatasink(Datasink):
    """
    Ray Data datasink for writing to FiftyOne datasets.

    This datasink enables distributed writing to FiftyOne datasets using
    Ray Data's parallel execution framework.

    Args:
        dataset_name: Name of the FiftyOne dataset to write to.
        filepath_field: Name of the column containing file paths (default: "filepath").
        label_fields: Optional mapping of field names to FiftyOne label types.
        overwrite: Whether to overwrite an existing dataset with the same name.
        persistent: Whether to make the dataset persistent.
        batch_size: Number of samples to add per batch.
        mongo_uri: Optional MongoDB URI if using a remote FiftyOne database.
        dataset_type: Optional media type hint ('image', 'video', etc.).

    Example:
        >>> sink = FiftyOneDatasink(
        ...     dataset_name="my_processed_dataset",
        ...     filepath_field="filepath",
        ...     label_fields={"predictions": "Detections"},
        ...     persistent=True,
        ... )
        >>> ds.write_datasink(sink)
    """

    def __init__(
        self,
        dataset_name: str,
        filepath_field: str = "filepath",
        label_fields: dict[str, str] | None = None,
        overwrite: bool = False,
        persistent: bool = True,
        batch_size: int = 100,
        mongo_uri: str | None = None,
        dataset_type: str | None = None,
    ) -> None:
        self._dataset_name = dataset_name
        self._filepath_field = filepath_field
        self._label_fields = label_fields or {}
        self._overwrite = overwrite
        self._persistent = persistent
        self._batch_size = batch_size
        self._mongo_uri = mongo_uri
        self._dataset_type = dataset_type

        # Statistics tracking
        self._total_samples_written = 0
        self._errors: list[str] = []

    def get_name(self) -> str:
        """Return a human-readable name for this datasink."""
        return f"FiftyOne({self._dataset_name})"

    @property
    def supports_distributed_writes(self) -> bool:
        """
        Indicate whether this sink supports distributed writes.

        FiftyOne's MongoDB backend supports concurrent writes, so we enable
        distributed writes.
        """
        return True

    @property
    def num_rows_per_write(self) -> int | None:
        """Target number of rows per write call."""
        return self._batch_size

    def on_write_start(self) -> None:
        """
        Initialize the dataset before writing.

        This is called once before any write tasks begin.
        """
        import fiftyone as fo

        # Configure MongoDB if specified
        if self._mongo_uri:
            fo.config.database_uri = self._mongo_uri

        # Check if dataset exists
        if fo.dataset_exists(self._dataset_name):
            if self._overwrite:
                fo.delete_dataset(self._dataset_name)
                logger.info(f"Deleted existing dataset: {self._dataset_name}")
            else:
                logger.info(f"Appending to existing dataset: {self._dataset_name}")
                return

        # Create new dataset
        dataset = fo.Dataset(name=self._dataset_name)
        dataset.persistent = self._persistent

        if self._dataset_type:
            dataset.media_type = self._dataset_type

        logger.info(f"Created dataset: {self._dataset_name}")

    def write(
        self,
        blocks: Iterable[Block],
        ctx: TaskContext,
    ) -> Any:
        """
        Write blocks of data to the FiftyOne dataset.

        Args:
            blocks: Iterable of PyArrow Table blocks to write.
            ctx: Task context from Ray Data.

        Returns:
            Write result statistics.
        """
        import fiftyone as fo

        # Configure MongoDB if specified
        if self._mongo_uri:
            fo.config.database_uri = self._mongo_uri

        # Load the dataset
        dataset = fo.load_dataset(self._dataset_name)

        samples_written = 0
        errors: list[str] = []

        for block in blocks:
            # Convert PyArrow table to list of dicts
            if isinstance(block, pa.Table):
                rows = block.to_pylist()
            else:
                # Handle pandas DataFrame
                rows = block.to_dict("records")

            # Create samples in batches
            samples: list[Any] = []

            for row in rows:
                try:
                    # Map filepath field if needed
                    if self._filepath_field != "filepath":
                        row["filepath"] = row.pop(self._filepath_field)

                    sample = helpers._dict_to_sample(row, self._label_fields)
                    samples.append(sample)

                    # Add batch when full
                    if len(samples) >= self._batch_size:
                        dataset.add_samples(samples)
                        samples_written += len(samples)
                        samples = []

                except Exception as e:
                    error_msg = f"Error creating sample: {e}"
                    errors.append(error_msg)
                    logger.warning(error_msg)

            # Add remaining samples
            if samples:
                dataset.add_samples(samples)
                samples_written += len(samples)

        return {
            "samples_written": samples_written,
            "errors": errors,
            "task_index": ctx.task_idx,
        }

    def on_write_complete(self, write_results: Any) -> None:
        """
        Finalize the dataset after all writes complete.

        Args:
            write_results: WriteResult object or results from all write tasks.
        """
        import fiftyone as fo

        total_written = 0
        all_errors: list[str] = []

        # Handle WriteResult object (newer Ray Data API)
        if hasattr(write_results, "rows_written"):
            total_written = write_results.rows_written or 0
        # Handle list of results (older API or custom implementation)
        elif isinstance(write_results, list):
            for result in write_results:
                if isinstance(result, dict):
                    total_written += result.get("samples_written", 0)
                    all_errors.extend(result.get("errors", []))
        # Handle single dict result
        elif isinstance(write_results, dict):
            total_written = write_results.get("samples_written", 0)
            all_errors.extend(write_results.get("errors", []))

        self._total_samples_written = total_written
        self._errors = all_errors

        # Configure MongoDB if specified
        if self._mongo_uri:
            fo.config.database_uri = self._mongo_uri

        # Log final statistics
        dataset = fo.load_dataset(self._dataset_name)
        logger.info(
            f"Write complete: {total_written} samples written to "
            f"'{self._dataset_name}' (total: {len(dataset)} samples)"
        )

        if all_errors:
            logger.warning(f"Encountered {len(all_errors)} errors during write")

    def on_write_failed(self, error: Exception) -> None:
        """
        Handle write failure.

        Args:
            error: The exception that caused the failure.
        """
        logger.error(f"Write failed for dataset '{self._dataset_name}': {error}")


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------


def write_fiftyone(
    dataset: ray.data.Dataset,
    dataset_name: str,
    filepath_field: str = "filepath",
    label_fields: dict[str, str] | None = None,
    overwrite: bool = False,
    persistent: bool = True,
    concurrency: int | None = None,
    **kwargs: Any,
) -> None:
    """
    Write a Ray Dataset to a FiftyOne dataset.

    This is a convenience function that creates a FiftyOneDatasink and
    calls dataset.write_datasink().

    Args:
        dataset: The Ray Dataset to write.
        dataset_name: Name for the output FiftyOne dataset.
        filepath_field: Name of the column containing file paths.
        label_fields: Optional mapping of field names to FiftyOne label types.
        overwrite: Whether to overwrite an existing dataset.
        persistent: Whether to make the dataset persistent.
        concurrency: Maximum number of concurrent write tasks.
        **kwargs: Additional arguments passed to FiftyOneDatasink.

    Example:
        >>> write_fiftyone(
        ...     ds,
        ...     "my_processed_dataset",
        ...     label_fields={"predictions": "Detections"},
        ... )
    """
    sink = FiftyOneDatasink(
        dataset_name=dataset_name,
        filepath_field=filepath_field,
        label_fields=label_fields,
        overwrite=overwrite,
        persistent=persistent,
        **kwargs,
    )

    dataset.write_datasink(sink, concurrency=concurrency)


# -----------------------------------------------------------------------------
# Module exports
# -----------------------------------------------------------------------------

__all__ = [
    "FiftyOneDatasink",
    "write_fiftyone",
]
