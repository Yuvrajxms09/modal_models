# API Usage

## Endpoints

- Qwen: `https://your-workspace--qwen-image-edit-endpoint.modal.run`
- Cosmos: `https://your-workspace--cosmos-text2image-web-endpoint.modal.run`
- Higgs: `https://your-workspace--higgs-audio-web-endpoint.modal.run`

## Qwen Image Edit

### `/generate` - Direct response

```bash
curl -X POST https://your-workspace--qwen-image-edit-endpoint.modal.run/generate \
  -F "prompt=add a sunset" \
  -F "input_image_url=https://example.com/image.jpg" \
  --output output.png
```

**Parameters:**
- `prompt` (required) - Edit description
- `input_image` or `input_image_url` (required) - Input image
- `negative_prompt` - Default: " "
- `seed` - Default: 0
- `true_guidance_scale` - Default: 4.0
- `num_inference_steps` - Default: 50

**Response:** PNG binary

### `/generate-async` - Async with webhook

```bash
curl -X POST https://your-workspace--qwen-image-edit-endpoint.modal.run/generate-async \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "input_image_url": "https://example.com/image.jpg",
      "prompt": "add a sunset"
    },
    "webhook_url": "https://your-app.com/webhook"
  }'
```

**Response:** Task ID, results sent to webhook

## Cosmos Text2Image

### `/generate` - Direct response

```bash
curl -X POST https://your-workspace--cosmos-text2image-web-endpoint.modal.run/generate \
  -F "prompt=a beautiful landscape" \
  --output image.jpg
```

**Parameters:**
- `prompt` (required)
- `aspect_ratio` - Default: "16:9" (options: "1:1", "4:3", "3:4", "16:9", "9:16")
- `negative_prompt` - Default: ""
- `seed` - Default: null
- `num_inference_steps` - Default: 30

**Response:** JPEG binary

### `/generate-async` - Returns S3 URL

```bash
curl -X POST https://your-workspace--cosmos-text2image-web-endpoint.modal.run/generate-async \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful landscape"}'
```

## Higgs Audio TTS

### `/generate` - Direct response

```bash
curl -X POST https://your-workspace--higgs-audio-web-endpoint.modal.run/generate \
  -F "text=Hello, this is a test" \
  --output audio.wav
```

**Parameters:**
- `text` (required)
- `system_prompt` - Default: auto
- `reference_audio_base64` - For voice cloning
- `reference_text` - Transcript of reference audio
- `temperature` - Default: 1.0
- `top_p` - Default: 0.95
- `top_k` - Default: 50

**Response:** WAV binary

### `/generate-async` - Returns S3 URL

```bash
curl -X POST https://your-workspace--higgs-audio-web-endpoint.modal.run/generate-async \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test"}'
```

## Python Examples

```python
import requests

# Qwen - Direct
with open("input.jpg", "rb") as f:
    response = requests.post(
        "https://your-workspace--qwen-image-edit-endpoint.modal.run/generate",
        files={"input_image": f},
        data={"prompt": "add a sunset"}
    )
    with open("output.png", "wb") as out:
        out.write(response.content)

# Cosmos - Direct
response = requests.post(
    "https://your-workspace--cosmos-text2image-web-endpoint.modal.run/generate",
    data={"prompt": "a beautiful landscape"}
)
with open("image.jpg", "wb") as f:
    f.write(response.content)

# Higgs - Direct
response = requests.post(
    "https://your-workspace--higgs-audio-web-endpoint.modal.run/generate",
    data={"text": "Hello world"}
)
with open("audio.wav", "wb") as f:
    f.write(response.content)
```
