# Qwen Image Edit Modal Deployment

Qwen Image Edit model deployment on Modal.

## Files

- **`inference.py`** - Main inference script for deploying Qwen Image Edit API on Modal
- **`upload_models.py`** - Script to download models to Modal volume

## Setup

### 1. Upload Models to Modal Volume

Update `upload_models.py` with the correct Qwen Image Edit model ID from HuggingFace, then run:

```bash
modal run qwenimageedit/upload_models.py
```

### 2. Deploy Inference API

Deploy the inference API:

```bash
modal deploy qwenimageedit/inference.py
```

## Requirements

- NVIDIA NGC account (for base image)
- Modal secrets configured:
  - `nvidia-ngc-secret` (NGC_API_KEY)
  - `huggingface-secret` (HF_TOKEN)
  - `aws-s3-secrets` (AWS credentials)

## API Endpoints

Once deployed, the API will be available at:
`https://<your-username>--qwen-image-edit-endpoint.modal.run`

### Endpoints

- `POST /generate` - Edit image from prompt
- `POST /generate-async` - Edit image asynchronously with webhook callback
- `GET /health` - Health check
