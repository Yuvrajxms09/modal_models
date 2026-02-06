# GLM-OCR Modal Deployment

GLM-OCR for high-quality document parsing and OCR deployment on Modal.

**Note:** This deployment requires the original GLM-OCR repository to be cloned locally in the parent directory as `GLM-OCR`.

## Files

- **`inference.py`** - Main inference script for deploying GLM-OCR API on Modal using SGLang
- **`upload.py`** - Script to download GLM-OCR models from HuggingFace to Modal volume

## Setup

### 1. Clone Original Repository

Ensure you have the original GLM-OCR repository cloned in the parent directory:

```bash
git clone https://github.com/zai-org/GLM-OCR ../GLM-OCR
```

### 2. Upload Models to Modal Volume

Download the GLM-OCR models to the Modal volume:

```bash
modal run glm_ocr/upload.py
```

### 3. Deploy Inference API

Deploy the inference API:

```bash
modal deploy glm_ocr/inference.py
```

## API Usage

Once deployed, the API will be available at:
`https://<your-workspace>--glm-ocr-api.modal.run`

### Endpoints

- `POST /glmocr/parse` - Parse images from URLs or local files

### Examples

#### Parse image from URL
```bash
curl -X POST https://<workspace>--glm-ocr-api.modal.run/glmocr/parse \
  -H "Content-Type: application/json" \
  -d '{"images": ["https://example.com/image.jpg"]}'
```

#### Parse image from local file
```bash
curl -X POST https://<workspace>--glm-ocr-api.modal.run/glmocr/parse \
  -F "files=@/path/to/image.png"
```

## Requirements

- Modal account with GPU access (any gpu should work)
- Python 3.11+
- HuggingFace token (for downloading models)
- Modal secrets configured:
  - `huggingface-secret` (with `HF_TOKEN`)
