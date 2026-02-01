
Inference deployments for TTS and image generation models on modal.com Each model runs as a separate FastAPI service with GPU acceleration and persistent volume storage.

**Note:** These are hobbyist deployments I run in my free time. They're not production-ready - missing authentication, rate limiting, proper error handling, and other production features. 

## Models

- **qwen3_tts** - Qwen3-TTS with custom voice, voice design, and voice cloning
- **soprano_tts** - Soprano TTS text-to-speech
- **higgs_audio_v2** - Higgs Audio TTS with voice cloning
- **cosmos_predict2_t2i** - Cosmos Predict2 text-to-image generation
- **omnigen2** - OmniGen2 image generation (text2img, editing, in-context)
- **qwen_image_edit** - Qwen Image Edit for natural language image editing

## Setup

### Clone Repository

```bash
git clone <repository-url>

```

### Configure Secrets

Set up Modal secrets:
- `huggingface-secret` (HF_TOKEN)
- `nvidia-ngc-secret` (NGC_API_KEY) - Required for cosmos_predict2_t2i, higgs_audio_v2, qwen_image_edit
- `aws-s3-secrets` (AWS credentials) - Required for S3 upload endpoints

### Upload Models

Each model folder contains an `upload_models.py` script. Update model IDs in the script, then run:

```bash
modal run <model-folder>/upload_models.py
```

Example:
```bash
modal run qwen3_tts/upload_models.py
```

### Deploy

```bash
modal deploy <model-folder>/inference.py
```

Example:
```bash
modal deploy qwen3_tts/inference.py
```

## API Usage

Endpoints follow the pattern: `https://<your-workspace>--<endpoint-label>.modal.run`

### Qwen3-TTS

```bash
# Custom voice
curl -X POST https://<workspace>--qwen3-tts-web-endpoint.modal.run/v1/audio/speech/custom-voice \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "language": "English", "speaker": "Ryan"}' \
  --output speech.wav

# Voice design
curl -X POST https://<workspace>--qwen3-tts-web-endpoint.modal.run/v1/audio/speech/voice-design \
  -H "Content-Type: application/json" \
  -d '{"input": "Text here", "language": "English", "instruct": "Speak in a cheerful tone"}' \
  --output speech.wav

# Voice clone
curl -X POST https://<workspace>--qwen3-tts-web-endpoint.modal.run/v1/audio/speech/voice-clone \
  -H "Content-Type: application/json" \
  -d '{"input": "Text here", "ref_audio": "https://example.com/ref.wav", "ref_text": "Reference transcript"}' \
  --output speech.wav
```

### Soprano TTS

```bash
curl -X POST https://<workspace>--soprano-web-endpoint.modal.run/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world"}' \
  --output speech.wav
```

### Cosmos Text2Image

```bash
curl -X POST https://<workspace>--cosmos-text2image-web-endpoint.modal.run/generate \
  -F "prompt=a beautiful landscape" \
  -F "aspect_ratio=16:9" \
  --output image.jpg
```

### OmniGen2

```bash
curl -X POST https://<workspace>--omnigen2.modal.run \
  -F "prompt=a beautiful landscape" \
  -F "width=1024" \
  -F "height=1024" \
  --output image.png
```

### Qwen Image Edit

```bash
curl -X POST https://<workspace>--qwen-image-edit-endpoint.modal.run/generate \
  -F "prompt=add a sunset" \
  -F "input_image_url=https://example.com/image.jpg" \
  --output output.png
```

### Higgs Audio

```bash
curl -X POST https://<workspace>--higgs-audio-web-endpoint.modal.run/generate \
  -F "text=Hello world" \
  --output audio.wav
```

## Caveats

- **No authentication** - endpoints are publicly accessible
- **No rate limiting** - can be abused
- **No monitoring/alerting** - failures go unnoticed
- **Basic error handling** - errors might not be user-friendly
- **No request validation** - malformed requests can crash services

These work fine for personal projects and testing, but don't use them for anything that needs reliability or security.

## Requirements

- Modal account with GPU access
- Python 3.11+
- Modal CLI installed (`pip install modal`)
