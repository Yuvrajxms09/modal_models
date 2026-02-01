# Soprano TTS Modal Deployment

Soprano TTS text-to-speech model deployment on Modal.

## Files

- **`inference.py`** - Main inference script for deploying Soprano TTS API on Modal
- **`upload_models.py`** - Script to download models to Modal volume (optional - Soprano loads from HuggingFace at runtime)

## Setup

### 1. Upload Models to Modal Volume (Optional)

**Note:** Soprano TTS loads models directly from HuggingFace at runtime, so uploading to volume is optional. If you want to pre-download models, update `upload_models.py` with the correct model ID and run:

```bash
modal run sopranostts/upload_models.py
```

### 2. Deploy Inference API

Deploy the inference API:

```bash
modal deploy sopranostts/inference.py
```

## API Endpoints

Once deployed, the API will be available at:
`https://<your-username>--soprano-web-endpoint.modal.run`

### Endpoints

- `POST /v1/audio/speech` - Generate speech from text
- `GET /health` - Health check

## Usage Example

```bash
curl -X POST https://<your-username>--soprano-web-endpoint.modal.run/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test.",
    "temperature": 0.3,
    "top_p": 0.95,
    "repetition_penalty": 1.2
  }' \
  --output speech.wav
```
