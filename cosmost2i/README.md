# Cosmos Predict2 Text2Image Modal Deployment

Cosmos Predict2 text-to-image generation model deployment on Modal.

## Files

- **`inference.py`** - Main inference script for deploying Cosmos Predict2 API on Modal
- **`upload_models.py`** - Script to download models to Modal volume

## Setup

### 1. Upload Models to Modal Volume

**Note:** Cosmos Predict2 models are typically downloaded from NVIDIA NGC. Update `upload_models.py` with the correct model IDs or download paths, then run:

```bash
modal run cosmost2i/upload_models.py
```

### 2. Deploy Inference API

Deploy the inference API:

```bash
modal deploy cosmost2i/inference.py
```

## Requirements

- NVIDIA NGC account (for model access)
- Modal secrets configured:
  - `nvidia-ngc-secret` (NGC_API_KEY)
  - `aws-s3-secrets` (AWS credentials)

## API Endpoints

Once deployed, the API will be available at:
`https://<your-username>--cosmos-text2image-web-endpoint.modal.run`

### Endpoints

- `POST /generate` - Generate image from text prompt
- `POST /generate-async` - Generate image and upload to S3
- `GET /health` - Health check
