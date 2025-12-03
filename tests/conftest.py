"""Shared test fixtures and configuration."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import pyarrow as pa


@pytest.fixture
def mock_fiftyone_sample():
    """Create a mock FiftyOne sample."""
    sample = Mock()
    sample.id = "sample123"
    sample.filepath = "/path/to/image.jpg"
    sample.to_dict.return_value = {
        "filepath": "/path/to/image.jpg",
        "id": "sample123",
        "tags": ["train"],
        "metadata": None,
    }
    return sample


@pytest.fixture
def mock_fiftyone_dataset():
    """Create a mock FiftyOne dataset."""
    dataset = Mock()
    dataset.name = "test_dataset"
    dataset.values.return_value = ["sample1", "sample2", "sample3"]
    dataset.__len__.return_value = 3
    dataset.__getitem__ = Mock()
    return dataset


@pytest.fixture
def mock_fiftyone_label():
    """Create a mock FiftyOne label with serialization support."""
    label = Mock()
    label.to_dict.return_value = {
        "_cls": "Classification",
        "label": "cat",
        "confidence": 0.95,
    }
    return label


@pytest.fixture
def sample_pyarrow_table():
    """Create a sample PyArrow table for testing."""
    data = {
        "filepath": ["/path/to/img1.jpg", "/path/to/img2.jpg"],
        "id": ["sample1", "sample2"],
        "tags": [["train"], ["val"]],
    }
    return pa.Table.from_pydict(data)


@pytest.fixture
def mock_ray_task_context():
    """Create a mock Ray TaskContext."""
    ctx = Mock()
    ctx.task_idx = 0
    return ctx


@pytest.fixture
def sample_label_dict():
    """Sample label dictionary with classification data."""
    return {
        "_cls": "Classification",
        "label": "dog",
        "confidence": 0.85,
    }


@pytest.fixture
def sample_detection_dict():
    """Sample detection dictionary."""
    return {
        "_cls": "Detections",
        "detections": [
            {
                "label": "person",
                "bounding_box": [0.1, 0.2, 0.3, 0.4],
                "confidence": 0.9,
            }
        ],
    }
