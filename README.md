# ray-fiftyone

Ray Data integration for FiftyOne datasets, enabling distributed processing of computer vision datasets using Ray's parallel execution framework.

## Overview

`ray-fiftyone` provides integration between [Ray Data](https://docs.ray.io/en/latest/data/data.html) and [FiftyOne](https://voxel51.com/fiftyone/), allowing you to:

- Read FiftyOne datasets into Ray Data for distributed processing
- Write Ray Data results back to FiftyOne datasets
- Leverage Ray's parallelization for large-scale computer vision workflows
- Export FiftyOne datasets to formats like Parquet for data warehousing

## Features

- **Distributed Reading**: Parallel loading of FiftyOne datasets with configurable parallelism
- **Distributed Writing**: Concurrent writes to FiftyOne datasets from Ray Data
- **Cloud Storage Support**: Works seamlessly with S3, Google Cloud Storage, and Azure Blob Storage
- **Remote MongoDB**: Connect to FiftyOne databases hosted in MongoDB Atlas or self-hosted instances
- **Field Selection**: Choose specific fields to include when reading datasets
- **Label Serialization**: Automatic serialization/deserialization of FiftyOne label types
- **View Stages**: Apply FiftyOne view stages (filtering, sorting) during data loading
- **Batch Processing**: Configurable batch sizes for optimal performance

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for package management. First, install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Basic Installation

Install the core package:

```bash
# Clone the repository
git clone <your-repo-url>
cd ray-fiftyone

# Install in editable mode
uv pip install -e .
```

### Installation with Optional Features

```bash
# Install with cloud storage support (S3, GCS, Azure)
uv pip install -e ".[cloud]"

# Install with ML dependencies (PyTorch, Transformers)
uv pip install -e ".[ml]"

# Install with test dependencies
uv pip install -e ".[test]"

# Install everything
uv pip install -e ".[all]"

# Combine multiple extras
uv pip install -e ".[cloud,ml]"
```

### Development Setup

For development, install development tools and pre-commit hooks:

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Install git hooks
pre-commit install

# (Optional) Run against all files
pre-commit run --all-files
```

## Quick Start

### Example 1: Export FiftyOne Dataset to Parquet

The included example script demonstrates reading a FiftyOne dataset and exporting it to Parquet format:

```bash
uv run python examples/save_dataset_to_parquet.py
```

This script:
1. Loads the FiftyOne "quickstart" dataset
2. Reads it into Ray Data using the `FiftyOneDatasource`
3. Flattens complex nested structures for Parquet compatibility
4. Exports to `/tmp/quickstart_parquet`

**Output Schema:**

```
Column                 Type
------                 ----
filepath               string
tags                   list<item: string>
metadata               null
created_at_json        string
last_modified_at_json  string
ground_truth_json      string
uniqueness             double
predictions_json       string
id                     string
```

The script automatically serializes complex FiftyOne label objects (detections, classifications, etc.) to JSON strings for Parquet compatibility.

### Example 2: Process and Write Back to FiftyOne

This example demonstrates a complete workflow: reading from FiftyOne, processing with Ray Data, and writing results back to FiftyOne:

```bash
uv run python examples/write_to_fiftyone.py
```

This script:
1. Reads the FiftyOne "quickstart" dataset
2. Limits to the first 50 samples using Ray Data
3. Writes the results to a new FiftyOne dataset called "quickstart_filtered"
4. Verifies the write and displays statistics

This pattern is useful for:
- Running distributed inference and storing predictions
- Processing and transforming large datasets
- Enriching datasets with embeddings or metadata
- Creating curated subsets for model training

### Example 3: Object Detection with DETR

Production-ready example using a pre-trained DETR model:

```bash
# Install ML dependencies
uv pip install -e ".[ml]"

# For cloud storage support, also install:
uv pip install -e ".[cloud]"

# Or install both at once:
uv pip install -e ".[ml,cloud]"

# Run detection
uv run python examples/object_detection_example.py
```

This script shows a complete ML workflow:
- Using Hugging Face's DETR model for object detection
- Processing images in batches for efficiency
- Converting model outputs to FiftyOne format
- Storing predictions with confidence scores and bounding boxes

This pattern can be adapted for any detection, segmentation, or classification model.

**Cloud Support:** All examples support cloud-hosted datasets. See [examples/CLOUD_SETUP.md](examples/CLOUD_SETUP.md) for configuration details.

## Usage

### Reading from FiftyOne

```python
import ray
from ray_fiftyone.data_source import FiftyOneDatasource

# Create a datasource
source = FiftyOneDatasource(
    dataset_name="my_dataset",
    fields=["filepath", "ground_truth", "predictions"],  # Optional: specify fields
    batch_size=100,
)

# Read into Ray Dataset
ds = ray.data.read_datasource(source)

# Process with Ray Data
ds = ds.map(lambda x: process_sample(x))
ds.show(5)
```

**Convenience function:**

```python
from ray_fiftyone.data_source import read_fiftyone

ds = read_fiftyone(
    "my_dataset",
    fields=["filepath", "ground_truth"],
    parallelism=10,
)
```

### Writing to FiftyOne

```python
from ray_fiftyone.data_sink import FiftyOneDatasink

# Create a datasink
sink = FiftyOneDatasink(
    dataset_name="output_dataset",
    filepath_field="filepath",
    label_fields={"predictions": "Detections"},  # Map fields to FiftyOne label types
    overwrite=True,
    persistent=True,
)

# Write Ray Dataset to FiftyOne
ds.write_datasink(sink)
```

**Convenience function:**

```python
from ray_fiftyone.data_sink import write_fiftyone

write_fiftyone(
    ds,
    dataset_name="output_dataset",
    label_fields={"predictions": "Detections"},
    overwrite=True,
)
```

### Advanced: View Stages

Apply FiftyOne view stages during reading:

```python
source = FiftyOneDatasource(
    dataset_name="my_dataset",
    view_stages=[
        {"stage": "match", "filter": {"uniqueness": {"$gt": 0.5}}},
        {"stage": "limit", "limit": 1000},
    ],
)
```

## API Reference

### FiftyOneDatasource

**Parameters:**
- `dataset_name` (str): Name of the FiftyOne dataset to read
- `fields` (list[str] | None): Optional list of fields to include
- `view_stages` (list[dict] | None): Optional view stage configurations
- `batch_size` (int): Number of samples per batch (default: 100)
- `include_id` (bool): Include sample ID in output (default: True)
- `mongo_uri` (str | None): Optional MongoDB URI for remote databases

### FiftyOneDatasink

**Parameters:**
- `dataset_name` (str): Name of the FiftyOne dataset to write to
- `filepath_field` (str): Name of the column containing file paths (default: "filepath")
- `label_fields` (dict[str, str] | None): Mapping of field names to FiftyOne label types
- `overwrite` (bool): Whether to overwrite existing dataset (default: False)
- `persistent` (bool): Whether to make the dataset persistent (default: True)
- `batch_size` (int): Number of samples to add per batch (default: 100)
- `mongo_uri` (str | None): Optional MongoDB URI for remote databases
- `dataset_type` (str | None): Optional media type hint ('image', 'video', etc.)

## Project Structure

```
ray-fiftyone/
├── ray_fiftyone/
│   ├── __init__.py
│   ├── data_source.py    # FiftyOneDatasource for reading
│   ├── data_sink.py      # FiftyOneDatasink for writing
│   └── helpers.py        # Serialization utilities
├── examples/
│   ├── save_dataset_to_parquet.py  # Export to Parquet example
│   └── write_to_fiftyone.py        # Write back to FiftyOne example
├── pyproject.toml
├── .pre-commit-config.yaml
└── README.md
```

## Requirements

### Core Dependencies

- Python >= 3.12
- fiftyone >= 1.10.0
- ray[data] >= 2.52.1
- pyarrow >= 22.0.0

### Optional Dependencies

Install using extras (e.g., `uv pip install -e ".[cloud]"`):

**Development** (`[dev]`):
- pre-commit >= 4.5.0
- ruff >= 0.1.0

**Testing** (`[test]`):
- pytest >= 8.0.0
- pytest-cov >= 4.1.0
- pytest-mock >= 3.12.0

**Cloud Storage** (`[cloud]`):
- s3fs >= 2024.0.0 (AWS S3)
- gcsfs >= 2024.0.0 (Google Cloud Storage)
- adlfs >= 2024.0.0 (Azure Blob Storage)
- fsspec >= 2024.0.0 (Unified filesystem)
- requests >= 2.31.0 (HTTP/HTTPS)

**Machine Learning** (`[ml]`):
- torch >= 2.0.0
- torchvision >= 0.15.0
- transformers >= 4.30.0
- pillow >= 10.0.0

**All Features** (`[all]`):
- Installs dev, test, cloud, and ml dependencies

## Testing

This project includes a comprehensive test suite with high code coverage.

### Running Tests

Run the full test suite:

```bash
uv run pytest tests/
```

Run tests with verbose output:

```bash
uv run pytest tests/ -v
```

Run tests with coverage report:

```bash
uv run pytest tests/ --cov=ray_fiftyone --cov-report=term-missing
```

Generate HTML coverage report:

```bash
uv run pytest tests/ --cov=ray_fiftyone --cov-report=html
# Open htmlcov/index.html in your browser
```

### Coverage Goals

The project maintains high test coverage:
- **Overall**: ~74%
- **helpers.py**: ~94%
- **data_sink.py**: ~82%
- **data_source.py**: ~55%

Note: Some integration code that directly interfaces with FiftyOne and Ray internals is intentionally untested to avoid excessive mocking complexity. These components are validated through the example scripts and integration testing.

## Contributing

This project uses:
- **uv** for package management
- **pre-commit** for code quality (ruff, pyupgrade)
- **ruff** for linting and formatting
- **pytest** for testing with coverage tracking

Before submitting changes:

1. Install all dependencies: `uv pip install -e ".[all]"`
2. Ensure pre-commit hooks pass: `pre-commit run --all-files`
3. Run the test suite: `uv run pytest tests/ --cov=ray_fiftyone`
4. Test your changes with the example scripts
5. Update documentation as needed

## Related Projects

- [FiftyOne](https://github.com/voxel51/fiftyone) - Open-source tool for building high-quality datasets and computer vision models
- [Ray Data](https://docs.ray.io/en/latest/data/data.html) - Scalable data processing library
