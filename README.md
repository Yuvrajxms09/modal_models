## Overview

This repository provides scalable, GPU-powered inference services deployed on [Modal](https://modal.com) for:

- **Qwen Image Edit** - Advanced image editing using diffusion models
- **Cosmos Predict2 Text2Image** - High-quality text-to-image generation
- **Higgs Audio TTS** - Text-to-speech with voice cloning capabilities

All services are designed for production use with automatic scaling, persistent model storage via Modal volumes, and S3-based output hosting.

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd 11_bs
   ```

2. **Set up Modal secrets** in Modal dashboard:
   - `huggingface-secret` (HF_TOKEN)
   - `nvidia-ngc-secret` (NGC_API_KEY)
   - `aws-s3-secrets` (AWS credentials)

3. **Upload models to Modal volumes**
   ```bash
   modal run modal/scripts/upload_models.py
   ```
   > **Note:** Configure model IDs and volume names in `modal/scripts/upload_models.py` before running.

4. **Deploy services**
   ```bash
   modal deploy modal/models/qwen.py modal/models/cosmos.py modal/models/higgs.py
   ```

5. **Use the API endpoints** - See [USAGE.md](./USAGE.md)

## Services

- **Qwen Image Edit** - Image editing with natural language
- **Cosmos Predict2** - Text-to-image generation
- **Higgs Audio** - Text-to-speech with voice cloning

## Requirements

- Modal account
- NVIDIA NGC account
- AWS account (for S3)
- HuggingFace account
