import modal
import os
import uuid
from typing import Dict, Any, List, Optional

app = modal.App("qwen-image-edit")

volume = modal.Volume.from_name("QwenImage-assets", create_if_missing=True)

MODEL_PATH = "/models"
DEFAULT_MODEL_PATH = "/models/Qwen-Image-Edit"

qwen_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:25.01-py3",
        secret=modal.Secret.from_name("nvidia-ngc-secret")
    ).pip_install(
        "torch",
        "diffusers>=0.25.0",
        "transformers>=4.45.1",
        "accelerate>=0.26.0",
        "boto3==1.35.36",
        "Pillow",
        "numpy",
        "requests",
    )
    .add_local_dir("Qwen-Image", remote_path="/app/Qwen-Image", copy=True, ignore=[".git", "__pycache__", "*.pyc"])
)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "pydantic==2.5.0",
        "python-multipart==0.0.6",
        "boto3==1.38.31",
        "httpx==0.25.2",
        "requests",
        "Pillow",
    )
)

@app.cls(image=qwen_image, gpu="A100-80GB", scaledown_window=180, timeout=2400, volumes={MODEL_PATH: volume}, secrets=[modal.Secret.from_name("aws-s3-secrets")])
@modal.concurrent(max_inputs=3, target_inputs=2)
class QwenImageEditGPU:
    _is_loaded = False

    @modal.enter()
    def load_pipeline(self):
        import sys
        import os
        import torch
        
        sys.path.insert(0, "/app/Qwen-Image")

        if self._is_loaded:
            return
        
        from diffusers import QwenImageEditPipeline
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        self.pipe = QwenImageEditPipeline.from_pretrained(
            DEFAULT_MODEL_PATH,
            torch_dtype=self.dtype,
            local_files_only=True
        ).to(self.device)
        
        self._is_loaded = True

    def upload_to_s3(self, image_bytes: bytes, image_index: int = 0) -> str:
        import boto3
        import io
        
        s3 = boto3.client("s3")
        bucket_name = "xxxxxxxx"
        filename = f"qwen_image_{uuid.uuid4().hex[:12]}_{image_index}.png"
        s3_key = f"qwen-image/{filename}"
        
        s3.upload_fileobj(
            io.BytesIO(image_bytes),
            bucket_name,
            s3_key,
            ExtraArgs={'ContentType': 'image/png'}
        )
        
        return f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"

    def download_image_from_url(self, image_url: str) -> bytes:
        import requests
        import base64
        
        if image_url.startswith("data:image/"):
            _, encoded = image_url.split(",", 1)
            return base64.b64decode(encoded)
        
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        return response.content

    @modal.method()
    def edit_images_direct(self,
                          input_image_url: str,
                          prompt: str,
                          negative_prompt: str = " ",
                          seed: int = 0,
                          true_guidance_scale: float = 4.0,
                          randomize_seed: bool = False,
                          num_inference_steps: int = 50
                          ) -> bytes:
        import torch
        import random
        import numpy as np
        import io
        from PIL import Image
        
        image_data = self.download_image_from_url(input_image_url)
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        if randomize_seed:
            seed = random.randint(0, np.iinfo(np.int32).max)
        
        generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.cuda.device(self.device):
            images = self.pipe(
                image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                num_inference_steps=num_inference_steps,
                generator=generator,
                true_cfg_scale=true_guidance_scale,
                num_images_per_prompt=1
            ).images

        img_buffer = io.BytesIO()
        images[0].save(img_buffer, format='PNG')
        return img_buffer.getvalue()

    @modal.method()
    def edit_images_and_upload(self, 
                             input_image_url: str,
                             prompt: str,
                             negative_prompt: str = " ",
                             seed: int = 0,
                             true_guidance_scale: float = 4.0,
                             randomize_seed: bool = False,
                             num_inference_steps: int = 50,
                             num_images: int = 1
                             ) -> Dict[str, Any]:
        import torch
        import random
        import numpy as np
        import io
        from PIL import Image
        
        image_data = self.download_image_from_url(input_image_url)
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        if randomize_seed:
            seed = random.randint(0, np.iinfo(np.int32).max)
        
        generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.cuda.device(self.device):
            images = self.pipe(
                image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                num_inference_steps=num_inference_steps,
                generator=generator,
                true_cfg_scale=true_guidance_scale,
                num_images_per_prompt=num_images
            ).images

        image_urls = []
        
        for i, image in enumerate(images):
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            image_urls.append(self.upload_to_s3(img_buffer.getvalue(), i))
        
        return {
            "image_urls": image_urls,
            "seed": seed,
            "num_images": len(images),
            "status": "success"
        }

    @modal.method()
    def process_image_edit_async(self, task_id: str, input_image_url: str, prompt: str, webhook_url: str,
                                 negative_prompt: str = " ", seed: int = 0, true_guidance_scale: float = 4.0,
                                 randomize_seed: bool = False, num_inference_steps: int = 50,
                                 num_images: int = 1, request_start_time: float = None):
        try:
            result = self.edit_images_and_upload.remote(
                input_image_url=input_image_url,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                true_guidance_scale=true_guidance_scale,
                randomize_seed=randomize_seed,
                num_inference_steps=num_inference_steps,
                num_images=num_images
            )
            self.send_webhook_callback.remote(webhook_url, task_id, "succeeded", result["image_urls"], request_start_time=request_start_time)
        except Exception as e:
            error_msg = str(e)
            if error_msg.startswith("400: "):
                error_msg = error_msg[5:]
            elif error_msg.startswith("500: "):
                error_msg = error_msg[5:]
            self.send_webhook_callback.remote(webhook_url, task_id, "error", error_msg=error_msg, request_start_time=request_start_time)
    
    @modal.method()
    def send_webhook_callback(self, webhook_url: str, task_id: str, status: str, output: Optional[List[str]] = None, error_msg: Optional[str] = None, request_start_time: float = None):
        import httpx
        import asyncio
        
        if status == "succeeded":
            payload = {"status": "succeeded", "exec_id": task_id, "output": output}
        else:
            payload = {"status": "error", "exec_id": task_id, "msg": error_msg or "Processing failed"}
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def send_webhook():
            async with httpx.AsyncClient() as client:
                await client.post(webhook_url, json=payload, timeout=30)
        
        loop.run_until_complete(send_webhook())
        loop.close()


@app.function(image=web_image, timeout=2400)
@modal.asgi_app(label="qwen-image-edit-endpoint")
def fastapi_app():
    from fastapi import FastAPI, Request, UploadFile, File, Form
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel, Field
    from datetime import datetime
    import uuid
    import base64
    import io
    
    class ModelRequest(BaseModel):
        inputs: Dict[str, Any]
        webhook_url: str
        model: Optional[str] = None
    
    web_app = FastAPI(title="Qwen Image Edit API", version="1.0.0")

    @web_app.post("/generate")
    async def generate_endpoint(
        prompt: str = Form(...),
        input_image: UploadFile = File(None),
        input_image_url: str = Form(None),
        negative_prompt: str = Form(" "),
        seed: int = Form(0),
        true_guidance_scale: float = Form(4.0),
        randomize_seed: bool = Form(False),
        num_inference_steps: int = Form(50)
    ):
        try:
            image_url = input_image_url
            if input_image:
                image_data = await input_image.read()
                image_url = f"data:image/{input_image.content_type or 'jpeg'};base64,{base64.b64encode(image_data).decode()}"
            
            result_bytes = QwenImageEditGPU().edit_images_direct.remote(
                input_image_url=image_url,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                true_guidance_scale=true_guidance_scale,
                randomize_seed=randomize_seed,
                num_inference_steps=num_inference_steps
            )
            
            return Response(content=result_bytes, media_type="image/png")
        except Exception as e:
            error_msg = str(e)
            status_code = 400 if error_msg.startswith("400:") else 500
            message = error_msg[4:].strip() if error_msg.startswith(("400:", "500:")) else "Internal server error occurred. Please try again."
            
            return JSONResponse(
                status_code=status_code,
                content={"status": "error", "message": message}
            )

    @web_app.post("/generate-async")
    async def generate_async_endpoint(request: ModelRequest):
        try:
            inputs = request.inputs
            task_id = str(uuid.uuid4())
            
            QwenImageEditGPU().process_image_edit_async.spawn(
                task_id,
                inputs.get("input_image_url"),
                inputs.get("prompt"),
                request.webhook_url,
                inputs.get("negative_prompt", " "),
                inputs.get("seed", 0),
                inputs.get("true_guidance_scale", 4.0),
                inputs.get("randomize_seed", False),
                inputs.get("num_inference_steps", 50),
                inputs.get("num_images", 1),
                None
            )
            
            return {
                "status": "success",
                "task_id": task_id,
                "created_at": datetime.utcnow().isoformat() + "Z"
            }
        except Exception as e:
            error_msg = str(e)
            status_code = 400 if error_msg.startswith("400:") else 500
            message = error_msg[4:].strip() if error_msg.startswith(("400:", "500:")) else "Internal server error occurred. Please try again."
            
            return JSONResponse(
                status_code=status_code,
                content={"status": "error", "message": message}
            )

    @web_app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "service": "qwen-image-edit"}

    return web_app
