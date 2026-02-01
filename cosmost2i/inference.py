import modal
import os
import uuid
from typing import Optional, Any

app = modal.App("cosmos-predict2-text2image")

volume = modal.Volume.from_name("nvidia_cosmos_assets", create_if_missing=False)

MODEL_PATH = "/models"

cosmos_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.1",
        secret=modal.Secret.from_name("nvidia-ngc-secret")
    )
    .pip_install(
        "better-profanity==0.7.0",
        "boto3==1.38.31",
        "decord==0.6.0",
        "diffusers==0.33.1",
        "ftfy==6.3.1",
        "fvcore==0.1.5.post20221221",
        "huggingface-hub==0.32.4",
        "hydra-core==1.3.2",
        "imageio[pyav,ffmpeg]==2.37.0",
        "iopath==0.1.10",
        "ipdb==0.13.13",
        "loguru==0.7.3",
        "mediapy==1.2.4",
        "megatron-core==0.12.1",
        "modelscope==1.26.0",
        "natten==0.20.1",
        "nltk==3.9.1",
        "omegaconf==2.3.0",
        "opencv-python==4.11.0.86",
        "peft==0.15.2",
        "qwen-vl-utils[decord]==0.0.11",
        "retinaface-py==0.0.2",
        "scikit-image==0.25.2",
        "sentencepiece==0.2.0",
        "termcolor==3.1.0",
        "transformers==4.51.3",
        "webdataset==0.2.111",
    )
    .add_local_dir("cosmos-predict2", remote_path="/app/cosmos-predict2", copy=True, ignore=[".git", "__pycache__", "*.pyc"])
)


web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "pydantic==2.5.0",
        "python-multipart==0.0.6",
        "boto3==1.38.31",
    )
)


@app.cls(image=cosmos_image, gpu="A100", scaledown_window=180, timeout=600, volumes={MODEL_PATH: volume}, secrets=[modal.Secret.from_name("aws-s3-secrets")])
@modal.concurrent(max_inputs=4, target_inputs=3)
class CosmosText2ImageGPU:
    _is_loaded = False

    @modal.enter()
    def load_pipeline(self):
        import sys
        import os
        import torch
        from megatron.core import parallel_state
        from cosmos_predict2.configs.base.config_text2image import PREDICT2_TEXT2IMAGE_PIPELINE_14B
        from cosmos_predict2.pipelines.text2image import Text2ImagePipeline
        from imaginaire.utils import distributed, log, misc
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.chdir(MODEL_PATH)
        sys.path.insert(0, "/app/cosmos-predict2")
        
        if self._is_loaded:
            return
        
        config = PREDICT2_TEXT2IMAGE_PIPELINE_14B
        dit_path = "checkpoints/nvidia/Cosmos-Predict2-14B-Text2Image/model.pt"
        text_encoder_path = "checkpoints/google-t5/t5-11b"

        misc.set_random_seed(seed=0, by_rank=True)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        is_distributed = parallel_state.is_initialized() and torch.distributed.is_initialized()

        if is_distributed:
            from imaginaire.utils.distributed import get_rank
            rank = get_rank()
            if rank == 0:
                self.pipeline = Text2ImagePipeline.from_config(
                    config=config,
                    dit_path=dit_path,
                    device="cuda",
                    torch_dtype=torch.bfloat16,
                )
            else:
                self.pipeline = None
        else:
            self.pipeline = Text2ImagePipeline.from_config(
                config=config,
                dit_path=dit_path,
                text_encoder_path=text_encoder_path,
                device="cuda",
                torch_dtype=torch.bfloat16,
            )
        
        if hasattr(self.pipeline, 'dit_ema') and self.pipeline.dit_ema is not None:
            self.pipeline.dit = self.pipeline.dit_ema
        
        self._is_loaded = True

    def upload_to_s3(self, image_bytes: bytes, image_index: int = 0) -> str:
        import boto3
        import io
        
        s3 = boto3.client("s3")
        bucket_name = "xxxxxxx"
        s3_key = f"cosmos-predict2/cosmos_text2image_{uuid.uuid4().hex[:12]}_{image_index}.jpg"
        
        s3.upload_fileobj(
            io.BytesIO(image_bytes),
            bucket_name,
            s3_key,
            ExtraArgs={'ContentType': 'image/jpeg'}
        )
        
        return f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"

    def process_single_generation(self, prompt: str, negative_prompt: str, aspect_ratio: str, seed: int, use_cuda_graphs: bool, num_inference_steps: int = 30, benchmark: bool = False) -> Any:
        return self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            seed=seed,
            num_sampling_step=num_inference_steps,
            use_cuda_graphs=use_cuda_graphs,
        )

    @modal.method()
    def generate_image_direct(self,
                              prompt: str,
                              negative_prompt: str = "",
                              aspect_ratio: str = "16:9",
                              seed: int = None,
                              use_cuda_graphs: bool = True,
                              num_inference_steps: int = 30
                              ) -> bytes:
        import tempfile
        import os
        from imaginaire.utils.io import save_image_or_video
        
        image = self.process_single_generation(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
            num_inference_steps=num_inference_steps
        )
        
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        save_image_or_video(image, temp_path)
        
        with open(temp_path, "rb") as f:
            image_bytes = f.read()
        
        os.remove(temp_path)
        return image_bytes

    @modal.method()
    def generate_image_and_upload(self, 
                                 prompt: str, 
                                 negative_prompt: str = "", 
                                 aspect_ratio: str = "16:9",
                                 seed: int = None, 
                                 use_cuda_graphs: bool = True,
                                 num_inference_steps: int = 30
                                 ) -> str:
        import tempfile
        import os
        from imaginaire.utils.io import save_image_or_video
        
        image = self.process_single_generation(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
            num_inference_steps=num_inference_steps
        )
        
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        save_image_or_video(image, temp_path)
        
        with open(temp_path, "rb") as f:
            image_bytes = f.read()
        
        os.remove(temp_path)
        return self.upload_to_s3(image_bytes, 0)

@app.function(image=web_image)
@modal.asgi_app(label="cosmos-text2image-web-endpoint")
def fastapi_app():
    from fastapi import FastAPI, Form
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel, Field, ConfigDict
    
    class CosmosText2ImageRequestAPI(BaseModel):
        model_config = ConfigDict(protected_namespaces=())
        prompt: str
        negative_prompt: str = ""
        aspect_ratio: str = "16:9"
        seed: Optional[int] = None
        use_cuda_graphs: bool = True
        num_inference_steps: int = 30

    class CosmosText2ImageResponseAPI(BaseModel):
        url: str
        status: str = "success"
        message: str = "Image generated successfully"

    web_app = FastAPI(title="Cosmos Predict2 Text2Image API", version="1.0.0")

    @web_app.post("/generate")
    async def generate_images_endpoint(
        prompt: str,
        negative_prompt: str = "",
        aspect_ratio: str = "16:9",
        seed: Optional[int] = None,
        use_cuda_graphs: bool = True,
        num_inference_steps: int = 30
    ):
        try:
            result_bytes = CosmosText2ImageGPU().generate_image_direct.remote(
                prompt=prompt,
                negative_prompt=negative_prompt,
                aspect_ratio=aspect_ratio,
                seed=seed,
                use_cuda_graphs=use_cuda_graphs,
                num_inference_steps=num_inference_steps
            )
            return Response(content=result_bytes, media_type="image/jpeg")
        except Exception as e:
            error_msg = str(e)
            status_code = 400 if error_msg.startswith("400:") else 500
            message = error_msg[4:].strip() if error_msg.startswith(("400:", "500:")) else "Internal server error occurred. Please try again."
            
            return JSONResponse(
                status_code=status_code,
                content={"status": "error", "message": message}
            )

    @web_app.post("/generate-async", response_model=CosmosText2ImageResponseAPI)
    async def generate_images_async_endpoint(request_data: CosmosText2ImageRequestAPI):
        try:
            public_url = CosmosText2ImageGPU().generate_image_and_upload.remote(
                prompt=request_data.prompt,
                negative_prompt=request_data.negative_prompt,
                aspect_ratio=request_data.aspect_ratio,
                seed=request_data.seed,
                use_cuda_graphs=request_data.use_cuda_graphs,
                num_inference_steps=request_data.num_inference_steps
            )
            return CosmosText2ImageResponseAPI(url=public_url)
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
        return {"status": "healthy", "service": "cosmos-text2image"}

    return web_app
