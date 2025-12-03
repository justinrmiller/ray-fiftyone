# Cloud Setup Guide

This guide explains how to use ray-fiftyone with cloud-hosted datasets and remote storage.

## Overview

Ray-fiftyone supports:
- **Remote MongoDB**: Host your FiftyOne database in MongoDB Atlas or self-hosted MongoDB
- **Cloud Storage**: Images stored in S3, Google Cloud Storage, Azure Blob Storage
- **Mixed Setup**: Remote database with cloud or local images

## Quick Start

### 1. Install Cloud Storage Dependencies

```bash
# For AWS S3
uv pip install s3fs boto3

# For Google Cloud Storage
uv pip install gcsfs google-cloud-storage

# For Azure Blob Storage
uv pip install adlfs azure-storage-blob

# Or install all
uv pip install s3fs gcsfs adlfs
```

### 2. Configure Credentials

**AWS S3:**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Or use AWS CLI
aws configure
```

**Google Cloud Storage:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Or use gcloud
gcloud auth application-default login
```

**Azure Blob Storage:**
```bash
export AZURE_STORAGE_ACCOUNT=your_account
export AZURE_STORAGE_KEY=your_key

# Or use Azure CLI
az login
```

### 3. Configure Remote MongoDB (Optional)

**MongoDB Atlas:**
```bash
export FIFTYONE_DATABASE_URI="mongodb+srv://username:password@cluster.mongodb.net/fiftyone"
```

**Self-Hosted MongoDB:**
```bash
export FIFTYONE_DATABASE_URI="mongodb://username:password@host:27017/fiftyone"
```

## Usage Examples

### Example 1: Remote MongoDB + S3 Images

```bash
# Set environment variables
export FIFTYONE_DATABASE_URI="mongodb+srv://user:pass@cluster.mongodb.net/fiftyone"
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Run detection
python examples/object_detection_example.py \
  --dataset my_s3_dataset \
  --limit 100 \
  --parallelism 8
```

### Example 2: Local MongoDB + GCS Images

```bash
# Only need GCS credentials
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Run detection (MongoDB is local)
python examples/object_detection_example.py \
  --dataset my_gcs_dataset \
  --limit 50
```

### Example 3: Command Line Arguments

```bash
# Pass MongoDB URI via command line
python examples/object_detection_example.py \
  --dataset my_dataset \
  --mongo-uri "mongodb://user:pass@remote-host:27017/fiftyone" \
  --limit 200 \
  --parallelism 16 \
  --batch-size 8 \
  --output my_predictions
```

## Creating Cloud-Hosted Datasets

### Option 1: FiftyOne Dataset with S3 Images

```python
import fiftyone as fo

# Create dataset
dataset = fo.Dataset("my_s3_dataset")

# Add samples with S3 URIs
samples = [
    fo.Sample(filepath="s3://my-bucket/images/img1.jpg"),
    fo.Sample(filepath="s3://my-bucket/images/img2.jpg"),
]
dataset.add_samples(samples)
dataset.persistent = True
```

### Option 2: Upload Local Dataset to Cloud

```python
import fiftyone as fo
import boto3

# Load local dataset
dataset = fo.load_dataset("local_dataset")

# Upload images to S3 and update filepaths
s3 = boto3.client('s3')
bucket = "my-bucket"

for sample in dataset:
    local_path = sample.filepath
    s3_key = f"images/{sample.id}.jpg"

    # Upload to S3
    s3.upload_file(local_path, bucket, s3_key)

    # Update filepath
    sample.filepath = f"s3://{bucket}/{s3_key}"
    sample.save()

print(f"Uploaded {len(dataset)} images to S3")
```

### Option 3: Use FiftyOne's MongoDB Atlas Integration

```python
import fiftyone as fo

# Configure remote database
fo.config.database_uri = "mongodb+srv://user:pass@cluster.mongodb.net/fiftyone"

# Now all datasets are stored remotely
dataset = fo.Dataset("remote_dataset")
# ... add samples ...
dataset.persistent = True
```

## Performance Optimization

### 1. Ray Cluster on Cloud

Run Ray on a cloud cluster for maximum parallelism:

```bash
# On head node
ray start --head --port=6379

# On worker nodes
ray start --address=head-node-ip:6379

# Run your script
RAY_ADDRESS=head-node-ip:6379 python examples/object_detection_example.py \
  --parallelism 100
```

### 2. Optimize Parallelism

```bash
# Match parallelism to available cores
python examples/object_detection_example.py \
  --parallelism $(nproc) \
  --batch-size 16
```

### 3. Regional Affinity

Run Ray workers in the same region as your cloud storage:
- S3 + EC2 in us-east-1
- GCS + GCE in us-central1
- Azure Blob + Azure VM in same region

This minimizes data transfer latency and costs.

## Troubleshooting

### Issue: "Access Denied" or "Permission Denied"

**Solution:**
```bash
# Verify credentials are set
env | grep AWS
env | grep GOOGLE
env | grep AZURE

# Test access manually
aws s3 ls s3://my-bucket/  # AWS
gsutil ls gs://my-bucket/   # GCS
az storage blob list --account-name myaccount  # Azure
```

### Issue: "Connection timeout" with MongoDB

**Solution:**
```bash
# Check MongoDB URI format
echo $FIFTYONE_DATABASE_URI

# Test connection with mongo shell
mongosh "mongodb+srv://user:pass@cluster.mongodb.net/fiftyone"

# Ensure IP whitelist includes your Ray cluster IPs
```

### Issue: Slow performance

**Solutions:**
1. Increase parallelism: `--parallelism 16`
2. Increase batch size: `--batch-size 32`
3. Use Ray cluster in same region as data
4. Enable image caching in Ray workers
5. Use Ray's object store for intermediate results

### Issue: "Module not found: s3fs/gcsfs"

**Solution:**
```bash
# Install required cloud storage library
uv pip install s3fs gcsfs adlfs
```

## Security Best Practices

1. **Use IAM Roles** (recommended over keys):
   ```bash
   # AWS: Attach IAM role to EC2 instances
   # GCP: Use Workload Identity
   # Azure: Use Managed Identity
   ```

2. **Encrypt credentials**:
   ```bash
   # Use secret managers
   export AWS_ACCESS_KEY_ID=$(aws secretsmanager get-secret-value \
     --secret-id my-key --query SecretString --output text)
   ```

3. **Limit access scope**:
   - S3: Bucket-specific policies
   - MongoDB: Database-specific users
   - GCS: Service account with minimal permissions

4. **Use VPCs**:
   - Place Ray cluster and MongoDB in same VPC
   - Use VPC endpoints for S3/GCS access

## Cost Optimization

1. **Data Transfer**:
   - Process data in same region as storage
   - Use Ray's object store to avoid repeated downloads
   - Cache model weights across workers

2. **Storage**:
   - Use cheaper storage classes for infrequently accessed data
   - Compress images when possible
   - Clean up temporary results

3. **Compute**:
   - Use spot instances for Ray workers
   - Scale down when not in use
   - Right-size instances for workload

## Example: Complete Production Setup

```bash
#!/bin/bash
# setup-cloud-detection.sh

# 1. Install dependencies
uv pip install torch torchvision transformers s3fs

# 2. Configure credentials (use IAM role in production)
export AWS_ACCESS_KEY_ID=${AWS_KEY}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET}
export FIFTYONE_DATABASE_URI=${MONGO_URI}

# 3. Start Ray cluster
ray start --head --num-cpus=16 --object-store-memory=10000000000

# 4. Run detection with optimal settings
python examples/object_detection_example.py \
  --dataset production_dataset \
  --limit 10000 \
  --parallelism 16 \
  --batch-size 32 \
  --output production_predictions

# 5. Cleanup
ray stop
```

## Next Steps

- Scale to multi-node Ray clusters with `ray up`
- Implement model serving with Ray Serve
- Set up continuous processing pipelines
- Monitor costs with cloud provider tools
- Use Ray's autoscaling for dynamic workloads

For more information:
- [Ray on AWS](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/aws.html)
- [Ray on GCP](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/gcp.html)
- [FiftyOne Cloud Integration](https://docs.voxel51.com/user_guide/dataset_creation/datasets.html#remote-data)
