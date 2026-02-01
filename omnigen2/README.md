# OmniGen2 Modal Deployment

OmniGen2 image generation model deployment on Modal.

## Files

- **`inference.py`** - Main inference script for deploying OmniGen2 API on Modal
- **`upload_models.py`** - Script to download models to Modal volume

## Setup

### 1. Upload Models to Modal Volume

Update `upload_models.py` with the correct OmniGen2 model ID from HuggingFace, then run:

```bash
modal run omnigen2/upload_models.py
```

### 2. Deploy Inference API

Deploy the inference API:

```bash
modal deploy omnigen2/inference.py
```

## Requirements

- Modal secrets configured:
  - `huggingface-secret` (HF_TOKEN)

## API Endpoints

Once deployed, the API will be available at:
`https://<your-username>--omnigen2.modal.run`

### Endpoints

- `POST /` - Generate images (supports text2img, editing, in_context)
- See inference.py for full parameter list
