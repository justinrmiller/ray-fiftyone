"""
Example: Distributed Object Detection with Ray and FiftyOne

This script demonstrates how to:
1. Load the FiftyOne quickstart dataset (local or cloud-hosted)
2. Run distributed object detection using Ray Data
3. Write predictions back to FiftyOne

Requirements:
    pip install torch torchvision transformers pillow

    # For cloud storage support:
    pip install s3fs gcsfs adlfs  # AWS S3, Google Cloud Storage, Azure

This example uses a pre-trained DETR (DEtection TRansformer) model from Hugging Face.

Environment Variables (for cloud datasets):
    FIFTYONE_DATABASE_URI: MongoDB connection string for remote FiftyOne database
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY: For S3 access
    GOOGLE_APPLICATION_CREDENTIALS: For GCS access
"""

import os
import argparse
import fiftyone as fo
import ray

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

from ray_fiftyone.data_source import read_fiftyone
from ray_fiftyone.data_sink import write_fiftyone


def load_image(filepath: str):
    """
    Load an image from local filesystem or cloud storage.

    Args:
        filepath: Local path or cloud URI (s3://, gs://, azure://, http://)

    Returns:
        PIL Image
    """
    from PIL import Image
    import io

    # Handle cloud URIs
    if filepath.startswith(("s3://", "gs://", "azure://", "az://", "adl://")):
        import fsspec

        # Open file using fsspec which handles cloud storage
        with fsspec.open(filepath, "rb") as f:
            image_bytes = f.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    elif filepath.startswith(("http://", "https://")):
        import requests

        response = requests.get(filepath)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        # Local file
        image = Image.open(filepath).convert("RGB")

    return image


@ray.remote
class ObjectDetectionActor:
    """Ray Actor for object detection with cached model state."""

    def __init__(self):
        """Initialize model on actor creation."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.model.eval()
        self.model.to(self.device)

    def detect_batch(self, batch: dict) -> dict:
        """Run object detection on a batch of images."""
        filepaths = batch["filepath"]
        predictions_list = []
        images = []
        valid_indices = []

        # Load all images first
        for idx, filepath in enumerate(filepaths):
            try:
                image = load_image(filepath)
                images.append(image)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                predictions_list.append({"_cls": "Detections", "detections": []})

        if not images:
            batch["predictions"] = predictions_list
            return batch

        # Batch process all images at once
        try:
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process all results
            target_sizes = torch.tensor([img.size[::-1] for img in images])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.7
            )

            # Convert results to FiftyOne format
            for img_idx, result in enumerate(results):
                detections = []
                for score, label_id, box in zip(
                    result["scores"], result["labels"], result["boxes"]
                ):
                    x_min, y_min, x_max, y_max = box.tolist()
                    img_width, img_height = images[img_idx].size

                    x = x_min / img_width
                    y = y_min / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    detection = {
                        "label": self.model.config.id2label[label_id.item()],
                        "bounding_box": [x, y, width, height],
                        "confidence": score.item(),
                    }
                    detections.append(detection)

                predictions_list.insert(
                    valid_indices[img_idx],
                    {"_cls": "Detections", "detections": detections},
                )

        except Exception as e:
            print(f"Error during batch inference: {e}")
            while len(predictions_list) < len(filepaths):
                predictions_list.append({"_cls": "Detections", "detections": []})

        batch["predictions"] = predictions_list
        return batch


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run distributed object detection on FiftyOne dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="quickstart",
        help="FiftyOne dataset name to process",
    )
    parser.add_argument(
        "--mongo-uri",
        type=str,
        default=None,
        help="MongoDB URI for remote FiftyOne database (e.g., mongodb://user:pass@host:27017)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of samples to process (default: 20)",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=4,
        help="Number of parallel Ray tasks (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output dataset name (default: <dataset>_with_predictions)",
    )

    args = parser.parse_args()

    print("Starting distributed object detection example...")
    print("Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Limit: {args.limit} samples")
    print(f"  Parallelism: {args.parallelism}")
    print(f"  Batch size: {args.batch_size}")
    if args.mongo_uri:
        print(f"  MongoDB: {args.mongo_uri.split('@')[-1]}")  # Hide credentials

    # Configure MongoDB URI if provided
    mongo_uri = args.mongo_uri or os.getenv("FIFTYONE_DATABASE_URI")
    if mongo_uri:
        fo.config.database_uri = mongo_uri
        print("\n✓ Connected to remote FiftyOne database")

    # Step 1: Load the FiftyOne dataset
    print(f"\n1. Loading FiftyOne dataset '{args.dataset}'...")

    if args.dataset == "quickstart" and not fo.dataset_exists("quickstart"):
        dataset = fo.zoo.load_zoo_dataset("quickstart")
        print(f"   Downloaded quickstart dataset with {len(dataset)} samples")
    else:
        dataset = fo.load_dataset(args.dataset)
        print(f"   Loaded dataset with {len(dataset)} samples")

    # Step 2: Initialize Ray (if not already initialized)
    print("\n2. Initializing Ray...")
    if not ray.is_initialized():
        ray.init()
    print("   Ray initialized successfully")

    # Step 3: Read dataset into Ray Data
    print("\n3. Reading dataset into Ray Data...")

    ray_dataset = read_fiftyone(
        dataset_name=args.dataset,
        fields=["filepath", "ground_truth"],  # Only read necessary fields
        view_stages=[{"stage": "limit", "limit": args.limit}],
        parallelism=args.parallelism,
        mongo_uri=mongo_uri,  # Pass mongo URI for remote access
    )

    print(f"   Created Ray Dataset with {ray_dataset.count()} samples")
    print("   Note: Dataset may contain cloud URIs (s3://, gs://, etc.)")

    # Step 4: Run distributed object detection
    print("\n4. Running distributed object detection...")
    print("   This will download the DETR model on first run (may take a few minutes)")
    print("   Model will be cached across workers for efficiency")

    # Create actor pool for batch processing
    num_actors = args.parallelism
    actors = [ObjectDetectionActor.remote() for _ in range(num_actors)]

    # Round-robin dispatch to actors
    actor_index = [0]  # Use list to maintain state in closure

    def process_batch(batch):
        actor = actors[actor_index[0] % len(actors)]
        actor_index[0] += 1
        return ray.get(actor.detect_batch.remote(batch))

    # Apply detection function to all samples
    detected_dataset = ray_dataset.map_batches(
        process_batch,
        batch_size=args.batch_size,
    )

    print("   Object detection complete")
    # Step 5: Write predictions back to FiftyOne
    print("\n5. Writing predictions to FiftyOne dataset...")
    output_dataset_name = args.output or f"{args.dataset}_with_predictions"

    # Delete existing output dataset if it exists
    if fo.dataset_exists(output_dataset_name):
        print(f"   Deleting existing dataset '{output_dataset_name}'")
        fo.delete_dataset(output_dataset_name)
        # Give FiftyOne time to clean up
        import time

        time.sleep(1)

    # Materialize Ray dataset before writing
    print("   Materializing predictions...")
    detected_data = detected_dataset.materialize()

    # Write to new dataset with predictions
    write_fiftyone(
        detected_data,
        dataset_name=output_dataset_name,
        label_fields={"predictions": "Detections"},
        overwrite=False,  # Changed to False since we already deleted
        persistent=True,
        mongo_uri=mongo_uri,
    )

    print(f"   Predictions written to '{output_dataset_name}' dataset")

    if mongo_uri:
        print("   Dataset stored in remote MongoDB")

    # Step 6: Display results
    print("\n6. Results Summary:")
    output_dataset = fo.load_dataset(output_dataset_name)
    print(f"   Dataset: {output_dataset_name}")
    print(f"   Total samples: {len(output_dataset)}")

    # Count samples with predictions
    samples_with_predictions = len(
        output_dataset.match(fo.ViewField("predictions.detections").length() > 0)
    )
    print(f"   Samples with detections: {samples_with_predictions}")

    # Show average number of detections per image
    total_detections = sum(
        len(sample.predictions.detections)
        for sample in output_dataset
        if sample.predictions
    )
    avg_detections = total_detections / len(output_dataset)
    print(f"   Average detections per image: {avg_detections:.2f}")

    # Step 7: Launch FiftyOne app to visualize
    print("\n7. Launching FiftyOne App...")
    print("   You can now view the predictions in the FiftyOne App")
    print("   Press Ctrl+C to exit when done viewing")

    session = fo.launch_app(output_dataset)
    session.wait()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting gracefully...")
        if ray.is_initialized():
            ray.shutdown()
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()

        if "AWS" in str(e) or "S3" in str(e):
            print("\nTip: For S3 access, ensure AWS credentials are configured:")
            print("  export AWS_ACCESS_KEY_ID=your_key")
            print("  export AWS_SECRET_ACCESS_KEY=your_secret")
            print("  or install and configure: aws configure")
        elif "GCS" in str(e) or "Google" in str(e):
            print("\nTip: For GCS access, set up application credentials:")
            print("  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json")

        if ray.is_initialized():
            ray.shutdown()
        raise
