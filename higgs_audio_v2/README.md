# Higgs Audio Modal Deployment

Higgs Audio text-to-speech model deployment on Modal.

## Files

- **`inference.py`** - Main inference script for deploying Higgs Audio API on Modal
- **`upload_models.py`** - Script to download models to Modal volume

## Setup

### 1. Upload Models to Modal Volume

Update `upload_models.py` with the correct Higgs Audio model IDs from HuggingFace, then run:

```bash
modal run higgs_audio_v2/upload_models.py
```

### 2. Deploy Inference API

Deploy the inference API:

```bash
modal deploy higgs_audio_v2/inference.py
```

## Requirements

- NVIDIA NGC account (for base image)
- Modal secrets configured:
  - `nvidia-ngc-secret` (NGC_API_KEY)
  - `huggingface-secret` (HF_TOKEN)
  - `aws-s3-secrets` (AWS credentials)

## API Endpoints

Once deployed, the API will be available at:
`https://<your-username>--higgs-audio-web-endpoint.modal.run`

### Endpoints

- `POST /generate` - Generate audio from text
- `POST /generate-async` - Generate audio and upload to S3
- `GET /health` - Health check
