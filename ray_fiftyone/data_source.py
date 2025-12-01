"""
Ray Data source for FiftyOne (Voxel51) datasets.

This module provides integration between Ray Data and FiftyOne, enabling
distributed processing of computer vision datasets using Ray's parallel
execution capabilities.

Usage:
    # Reading from FiftyOne
    ds = ray.data.read_datasource(
        FiftyOneDatasource(
            dataset_name="my_dataset",
            fields=["filepath", "ground_truth"],
        )
    )

"""

from __future__ import annotations

import logging
from typing import (
    Any,
)
from collections.abc import Callable, Iterator

import pyarrow as pa

import ray

from ray_fiftyone import helpers

from ray.data.block import Block, BlockMetadata
from ray.data.datasource import Datasource, ReadTask

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# FiftyOne Datasource (Reader)
# -----------------------------------------------------------------------------


class FiftyOneDatasource(Datasource):
    """
    Ray Data datasource for reading from FiftyOne datasets.

    This datasource enables distributed reading of FiftyOne datasets using
    Ray Data's parallel execution framework.

    Args:
        dataset_name: Name of the FiftyOne dataset to read.
        fields: Optional list of fields to include. If None, all fields are included.
        view_stages: Optional list of view stage configurations to apply
            (e.g., filtering, sorting). Each stage should be a dict with
            'stage' key and optional parameters.
        batch_size: Number of samples per batch when reading.
        include_id: Whether to include the sample ID in output rows.
        mongo_uri: Optional MongoDB URI if using a remote FiftyOne database.

    Example:
        >>> source = FiftyOneDatasource(
        ...     dataset_name="coco-2017-validation",
        ...     fields=["filepath", "ground_truth"],
        ...     batch_size=100,
        ... )
        >>> ds = ray.data.read_datasource(source)
    """

    def __init__(
        self,
        dataset_name: str,
        fields: list[str] | None = None,
        view_stages: list[dict[str, Any]] | None = None,
        batch_size: int = 100,
        include_id: bool = True,
        mongo_uri: str | None = None,
    ) -> None:
        self._dataset_name = dataset_name
        self._fields = fields
        self._view_stages = view_stages or []
        self._batch_size = batch_size
        self._include_id = include_id
        self._mongo_uri = mongo_uri

        # Cache dataset info for estimation
        self._num_samples: int | None = None

    def get_name(self) -> str:
        """Return a human-readable name for this datasource."""
        return f"FiftyOne({self._dataset_name})"

    def _get_dataset(self) -> Any:
        """
        Load and return the FiftyOne dataset.

        Returns:
            A FiftyOne Dataset or DatasetView instance.
        """
        import fiftyone as fo

        # Configure MongoDB if specified
        if self._mongo_uri:
            fo.config.database_uri = self._mongo_uri

        # Load the dataset
        dataset = fo.load_dataset(self._dataset_name)

        # Apply view stages if any
        view = dataset
        for stage_config in self._view_stages:
            stage_name = stage_config.get("stage")
            stage_params = {k: v for k, v in stage_config.items() if k != "stage"}

            if hasattr(view, stage_name):
                stage_method = getattr(view, stage_name)
                view = stage_method(**stage_params)

        return view

    def _get_sample_ids(self) -> list[str]:
        """
        Get all sample IDs from the dataset.

        Returns:
            List of sample ID strings.
        """
        dataset = self._get_dataset()
        return list(dataset.values("id"))

    def estimate_inmemory_data_size(self) -> int | None:
        """
        Estimate the in-memory data size.

        Returns:
            Estimated size in bytes, or None if unknown.
        """
        try:
            dataset = self._get_dataset()
            self._num_samples = len(dataset)

            # Rough estimate: ~10KB per sample (varies significantly by content)
            return self._num_samples * 10 * 1024
        except Exception as e:
            logger.warning(f"Could not estimate data size: {e}")
            return None

    def get_read_tasks(self, parallelism: int, **kwargs: Any) -> list[ReadTask]:
        """
        Generate read tasks for parallel execution.

        Args:
            parallelism: Requested number of parallel tasks.

        Returns:
            List of ReadTask instances.
        """
        # Get sample IDs
        sample_ids = self._get_sample_ids()
        num_samples = len(sample_ids)

        if num_samples == 0:
            return []

        # Calculate partition sizes
        num_tasks = min(parallelism, num_samples)
        samples_per_task = (num_samples + num_tasks - 1) // num_tasks

        read_tasks: list[ReadTask] = []

        for i in range(num_tasks):
            start_idx = i * samples_per_task
            end_idx = min((i + 1) * samples_per_task, num_samples)

            if start_idx >= num_samples:
                break

            partition_ids = sample_ids[start_idx:end_idx]
            partition_size = len(partition_ids)

            # Create metadata for this partition
            metadata = BlockMetadata(
                num_rows=partition_size,
                size_bytes=partition_size * 10 * 1024,  # Rough estimate
                input_files=None,
                exec_stats=None,
            )

            # Create the read function for this partition
            read_fn = self._create_read_fn(partition_ids)
            read_tasks.append(ReadTask(read_fn, metadata))

        return read_tasks

    def _create_read_fn(
        self,
        sample_ids: list[str],
    ) -> Callable[[], Iterator[Block]]:
        """
        Create a read function for a partition of samples.

        Args:
            sample_ids: List of sample IDs to read.

        Returns:
            A callable that yields PyArrow blocks.
        """
        dataset_name = self._dataset_name
        fields = self._fields
        include_id = self._include_id
        batch_size = self._batch_size
        mongo_uri = self._mongo_uri

        def read_partition() -> Iterator[pa.Table]:
            import fiftyone as fo

            # Configure MongoDB if specified
            if mongo_uri:
                fo.config.database_uri = mongo_uri

            # Load dataset
            dataset = fo.load_dataset(dataset_name)

            # Process samples in batches
            rows: list[dict[str, Any]] = []

            for sample_id in sample_ids:
                try:
                    sample = dataset[sample_id]
                    row = helpers._sample_to_dict(
                        sample,
                        fields=fields,
                        include_id=include_id,
                    )
                    rows.append(row)

                    # Yield batch when full
                    if len(rows) >= batch_size:
                        yield pa.Table.from_pylist(rows)
                        rows = []

                except Exception as e:
                    logger.warning(f"Error reading sample {sample_id}: {e}")
                    continue

            # Yield remaining rows
            if rows:
                yield pa.Table.from_pylist(rows)

        return read_partition


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------


def read_fiftyone(
    dataset_name: str,
    fields: list[str] | None = None,
    view_stages: list[dict[str, Any]] | None = None,
    parallelism: int = -1,
    **kwargs: Any,
) -> ray.data.Dataset:
    """
    Read a FiftyOne dataset into a Ray Dataset.

    This is a convenience function that creates a FiftyOneDatasource and
    calls ray.data.read_datasource().

    Args:
        dataset_name: Name of the FiftyOne dataset to read.
        fields: Optional list of fields to include.
        view_stages: Optional list of view stage configurations.
        parallelism: Number of parallel read tasks (-1 for auto).
        **kwargs: Additional arguments passed to FiftyOneDatasource.

    Returns:
        A Ray Dataset containing the FiftyOne data.

    Example:
        >>> ds = read_fiftyone(
        ...     "coco-2017-validation",
        ...     fields=["filepath", "ground_truth"],
        ... )
        >>> ds.show(5)
    """
    import ray

    source = FiftyOneDatasource(
        dataset_name=dataset_name,
        fields=fields,
        view_stages=view_stages,
        **kwargs,
    )

    return ray.data.read_datasource(
        source,
        parallelism=parallelism if parallelism > 0 else None,
    )


# -----------------------------------------------------------------------------
# Module exports
# -----------------------------------------------------------------------------

__all__ = [
    "FiftyOneDatasource",
    "read_fiftyone",
]
