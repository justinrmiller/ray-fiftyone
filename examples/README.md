# Examples

This directory contains example scripts demonstrating how to use ray-fiftyone for various computer vision workflows.

## Installation

Before running examples, install the required dependencies:

```bash
# Basic examples (1-2)
uv pip install -e .

# Object detection example (3) - requires ML dependencies
uv pip install -e ".[ml]"

# Cloud storage support (for any example with cloud datasets)
uv pip install -e ".[cloud]"

# Everything
uv pip install -e ".[all]"
```

## Quick Start Examples

### 1. Export to Parquet (`save_dataset_to_parquet.py`)

**What it does:** Exports a FiftyOne dataset to Parquet format for data warehousing.

**Use case:** Data engineering, ETL pipelines, archival storage.

```bash
uv run python examples/save_dataset_to_parquet.py
```

**Key features:**
- Reads FiftyOne datasets with Ray Data
- Flattens complex label structures to JSON
- Exports to Parquet format
- Demonstrates field selection

---

### 2. Filter and Write Back (`write_to_fiftyone.py`)

**What it does:** Reads a dataset, applies filters with Ray, writes back to FiftyOne.

**Use case:** Dataset curation, creating training/validation splits.

```bash
uv run python examples/write_to_fiftyone.py
```

**Key features:**
- Distributed data filtering
- Writing results to new FiftyOne datasets
- Verification and statistics

---

### 3. Object Detection with DETR (`object_detection_example.py`)

**What it does:** Production-ready object detection using Hugging Face's DETR model.

**Use case:** Real-world inference at scale, annotation assistance.

**Requirements:**
```bash
uv pip install torch torchvision transformers pillow

# For cloud storage support:
uv pip install s3fs gcsfs adlfs
```

**Run:**
```bash
# Local dataset
uv run python examples/object_detection_example.py

# Cloud dataset with options
python examples/object_detection_example.py \
  --dataset my_cloud_dataset \
  --mongo-uri "mongodb+srv://user:pass@cluster.mongodb.net/fiftyone" \
  --limit 100 \
  --parallelism 8 \
  --batch-size 16
```

**Key features:**
- Pre-trained DETR (DEtection TRansformer) model
- **Cloud storage support** (S3, GCS, Azure)
- **Remote MongoDB** for distributed teams
- Batch processing for efficiency
- Proper bounding box format conversion
- Confidence threshold filtering
- Full FiftyOne label integration
- Command-line configuration

**Command-line options:**
- `--dataset`: Dataset name to process
- `--mongo-uri`: Remote MongoDB connection string
- `--limit`: Number of samples to process
- `--parallelism`: Number of parallel Ray tasks
- `--batch-size`: Images per batch
- `--output`: Output dataset name

**Cloud setup:**
See [CLOUD_SETUP.md](./CLOUD_SETUP.md) for detailed instructions on:
- Configuring cloud storage credentials
- Setting up remote MongoDB
- Performance optimization
- Security best practices

**Adapting for your model:**
Replace the `run_object_detection()` function with your own model inference code. The key is to:
1. Load images from `batch["filepath"]` (handles cloud URIs automatically)
2. Run your model
3. Format predictions as FiftyOne label dictionaries
4. Return batch with predictions added

---

## Common Patterns

### Reading from FiftyOne
```python
from ray_fiftyone.data_source import read_fiftyone

ray_dataset = read_fiftyone(
    "my_dataset",
    fields=["filepath", "ground_truth"],  # Select specific fields
    view_stages=[{"stage": "limit", "limit": 100}],  # Apply FiftyOne view stages
    parallelism=4,  # Number of parallel read tasks
)
```

### Processing with Ray
```python
def process_batch(batch: dict) -> dict:
    # Your processing logic here
    batch["new_field"] = process(batch["filepath"])
    return batch

processed = ray_dataset.map_batches(
    process_batch,
    batch_size=10,
    num_cpus=1,
)
```

### Writing to FiftyOne
```python
from ray_fiftyone.data_sink import write_fiftyone

write_fiftyone(
    processed,
    dataset_name="output_dataset",
    label_fields={"predictions": "Detections"},  # Map fields to FiftyOne types
    overwrite=True,
)
```

---

## Label Type Reference

When writing predictions back to FiftyOne, specify the label type:

| FiftyOne Type | Use Case | Example |
|--------------|----------|---------|
| `Classification` | Single label per image | Image classification |
| `Classifications` | Multiple labels per image | Multi-label classification |
| `Detections` | Bounding boxes | Object detection |
| `Polylines` | Line segments | Lane detection, pose |
| `Keypoints` | Point annotations | Facial landmarks |
| `Segmentation` | Pixel masks | Semantic segmentation |

Example format for Detections:
```python
{
    "_cls": "Detections",
    "detections": [
        {
            "label": "person",
            "bounding_box": [x, y, width, height],  # Normalized [0-1]
            "confidence": 0.95
        }
    ]
}
```

---

## Tips

1. **Start small:** Test with `.limit(10)` before processing full datasets
2. **Batch size:** Balance between throughput and memory (start with 10-20)
3. **Parallelism:** Set to number of CPU cores available
4. **Field selection:** Only read fields you need to reduce memory usage
5. **View stages:** Apply FiftyOne filters before reading to process less data

---

## Next Steps

- Adapt the object detection example for your own models
- Combine with Ray's distributed training for active learning loops
- Export curated datasets to Parquet for training pipelines
- Build annotation assistance workflows

For more information, see the main [README](../README.md).
