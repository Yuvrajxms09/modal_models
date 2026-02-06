import modal
import os

# Modal App Configuration
app = modal.App("glm-ocr-service")

# Storage Configuration
MODEL_VOLUME_NAME = "glm-ocr-assets"
MODEL_MOUNT_PATH = "/models"

# Optimized Image for GLM-OCR
glm_ocr_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git", "libgl1-mesa-glx", "libglib2.0-0", "libnuma-dev",
        "libsm6", "libxext6", "libxrender1"
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "opencv-python",
        "pypdfium2",
        "pydantic",
        "pyyaml",
        "requests",
        "fastapi",
        "uvicorn",
        "accelerate",
        "sentencepiece",
        "sgl-kernel>=0.0.3",
        "pillow",
        "flask",
        "wordfreq",
        "python-multipart",
    )
    # Force upgrade CuDNN to bypass the Torch 2.5.1 strict pinning conflict
    .run_commands("pip install --upgrade nvidia-cudnn-cu12==9.16.0.29")
    .run_commands(
        "pip install git+https://github.com/sgl-project/sglang.git#subdirectory=python",
        "pip install git+https://github.com/huggingface/transformers.git"
    )
    # Copy and install SDK properly
    .add_local_dir("../GLM-OCR", remote_path="/opt/glm-ocr-sdk", copy=True)
    # Install the SDK in a way that it's globally available
    .run_commands("cd /opt/glm-ocr-sdk && pip install .")
)

volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

@app.cls(
    image=glm_ocr_image,
    gpu="A100", 
    volumes={MODEL_MOUNT_PATH: volume},
    scaledown_window =300,
    timeout=1200,
)
class GLMOCRWorker:
    @modal.enter()
    def start_service(self):
        import subprocess
        import time
        import requests
        
        print("🚀 Starting SGLang server...")
        
        env = os.environ.copy()
        env["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
        env["HF_HOME"] = f"{MODEL_MOUNT_PATH}/huggingface"
        # Ensure the SDK is in the path for the server to find its modules
        env["PYTHONPATH"] = f"/opt/glm-ocr-sdk:{env.get('PYTHONPATH', '')}"
        
        sglang_cmd = [
            "python", "-m", "sglang.launch_server",
            "--model", "zai-org/GLM-OCR",
            "--port", "8080",
            "--trust-remote-code",
            "--served-model-name", "glm-ocr",
            "--speculative-algorithm", "NEXTN",
            "--speculative-num-steps", "3",
            "--speculative-eagle-topk", "1",
            "--speculative-num-draft-tokens", "4",
        ]
        
        self.sglang_process = subprocess.Popen(sglang_cmd, env=env)
        
        # SGLang Health check
        for i in range(150):
            try:
                if requests.get("http://127.0.0.1:8080/health").status_code == 200:
                    print("✅ SGLang is ready.")
                    break
            except:
                pass
            time.sleep(2)
        else:
            if self.sglang_process.poll() is not None:
                print(f"SGLang process exited with code {self.sglang_process.returncode}")
            raise RuntimeError("SGLang failed to start.")

        # 2. Start official GLM-OCR SDK server
        print("🚀 Starting GLM-OCR SDK server...")
        self.flask_process = subprocess.Popen([
            "python", "-m", "glmocr.server",
            "--config", "/opt/glm-ocr-sdk/glmocr/config.yaml"
        ], env=env)
        
        # Now we use the REAL health endpoint from the SDK
        for i in range(60):
            try:
                if requests.get("http://127.0.0.1:5002/health").status_code == 200:
                    print("✅ SDK server online.")
                    break
            except:
                pass
            time.sleep(2)
        else:
            if self.flask_process.poll() is not None:
                print(f"SDK server process exited with code {self.flask_process.returncode}")
            raise RuntimeError("SDK server failed to start.")

    @modal.exit()
    def stop_service(self):
        if hasattr(self, 'flask_process'): self.flask_process.terminate()
        if hasattr(self, 'sglang_process'): self.sglang_process.terminate()

    @modal.method()
    def parse(self, images: list):
        import requests
        import base64
        from io import BytesIO
        
        processed_images = []
        for img in images:
            if isinstance(img, str) and img.startswith(("http://", "https://")):
                try:
                    # Pre-download images that the SDK server can't handle directly
                    resp = requests.get(img, timeout=30)
                    resp.raise_for_status()
                    
                    # Convert to data URL so the SDK server's PageLoader can process it
                    content_type = resp.headers.get("Content-Type", "image/png")
                    b64_data = base64.b64encode(resp.content).decode("utf-8")
                    processed_images.append(f"data:{content_type};base64,{b64_data}")
                except Exception as e:
                    print(f"Failed to download image {img}: {e}")
                    processed_images.append(img) # Fallback to original
            else:
                processed_images.append(img)

        response = requests.post(
            "http://127.0.0.1:5002/glmocr/parse",
            json={"images": processed_images},
            timeout=600
        )
        if response.status_code != 200:
            raise Exception(f"SDK Error: {response.text}")
        return response.json()

@app.function(image=glm_ocr_image)
@modal.asgi_app(label="glm-ocr-api")
def fastapi_app():
    from fastapi import FastAPI, Request, UploadFile, File
    from typing import List, Optional
    import base64
    
    web_app = FastAPI(title="GLM-OCR API")

    @web_app.post("/glmocr/parse")
    async def parse_endpoint(
        request: Request, 
        files: Optional[List[UploadFile]] = File(None)
    ):
        # 1. Handle Multipart Form Data (Direct File Uploads)
        if files:
            images = []
            for file in files:
                content = await file.read()
                # Convert to base64 data URL
                b64_data = base64.b64encode(content).decode("utf-8")
                mime_type = file.content_type or "image/png"
                images.append(f"data:{mime_type};base64,{b64_data}")
            
            return GLMOCRWorker().parse.remote(images)

        # 2. Handle JSON Body (URLs or Base64 strings)
        try:
            data = await request.json()
            images = data.get("images", [])
            if not images:
                return {"error": "No images provided in JSON body"}, 400
            
            return GLMOCRWorker().parse.remote(images if isinstance(images, list) else [images])
        except Exception:
            return {"error": "Invalid request. Provide either files or a JSON body with 'images'."}, 400

    return web_app
