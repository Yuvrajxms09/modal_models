# Qwen3-TTS Modal Deployment

Qwen3-TTS text-to-speech model deployment on Modal.

## Files

- **`inference.py`** - Main inference script for deploying Qwen3-TTS API on Modal
- **`upload_models.py`** - Script to download Qwen3-TTS models from HuggingFace to Modal volume

## Setup

### 1. Upload Models to Modal Volume

First, download all Qwen3-TTS models to the Modal volume:

```bash
modal run qwen3/upload_models.py
```

This will download:
- Qwen3-TTS-Tokenizer-12Hz
- Qwen3-TTS-12Hz-1.7B-CustomVoice
- Qwen3-TTS-12Hz-1.7B-VoiceDesign
- Qwen3-TTS-12Hz-1.7B-Base
- Qwen3-TTS-12Hz-0.6B-CustomVoice
- Qwen3-TTS-12Hz-0.6B-Base

### 2. Deploy Inference API

Deploy the inference API:

```bash
modal deploy qwen3/inference.py
```

## Configuration

### Model Selection

Set the `QWEN3_TTS_MODEL` environment variable to specify which model to load:

```bash
# For CustomVoice model (default)
export QWEN3_TTS_MODEL=Qwen3-TTS-12Hz-1.7B-CustomVoice

# For VoiceDesign model
export QWEN3_TTS_MODEL=Qwen3-TTS-12Hz-1.7B-VoiceDesign

# For Base (Voice Clone) model
export QWEN3_TTS_MODEL=Qwen3-TTS-12Hz-1.7B-Base
```

## API Endpoints

Once deployed, the API will be available at:
`https://<your-username>--qwen3-tts-web-endpoint.modal.run`

### Endpoints

- `POST /v1/audio/speech/custom-voice` - Custom voice generation
- `POST /v1/audio/speech/voice-design` - Voice design generation
- `POST /v1/audio/speech/voice-clone` - Voice cloning
- `GET /health` - Health check
- `GET /model-info` - Model information
- `GET /supported-speakers` - List of supported speakers
- `GET /supported-languages` - List of supported languages

## Usage Examples

See the main repository README for detailed API usage examples and Postman test cases.

## Requirements

- Modal account with GPU access
- HuggingFace token (for downloading models)
- Modal secrets configured:
  - `huggingface-secret` (with `HF_TOKEN`)
