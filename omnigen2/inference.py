import time
import os
from pathlib import Path
from typing import Dict, Any, Annotated
import base64
import fastapi
import modal
from fastapi.responses import Response
from fastapi import HTTPException

app = modal.App("Omnigen2-Testing")

volume = modal.Volume.from_name("omnigen2-assets", create_if_missing=False)

MODEL_PATH = "/models"

# Using  official OmniGen2 installation instructions
base_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        extra_index_url="https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "timm",
        "einops", 
        "accelerate",
        "transformers==4.51.3",
        "diffusers",
        "opencv-python-headless",
        "scipy",
        "wandb",
        "matplotlib",
        "Pillow",
        "tqdm",
        "omegaconf",
        "python-dotenv",
        "ninja",
        "ipykernel",
        "wheel",
        "fastapi[standard]==0.115.8",
        "filetype"
    )
    .run_commands(
        "pip install flash-attn==2.7.4.post1 --no-build-isolation"
    )
    .add_local_dir("OmniGen2", remote_path="/app/omnigen2", copy=True, ignore=[".git", "__pycache__", "*.pyc"])
)


# one of replicate/fal is using L40S for this model
# we are currently using A100-40GB version
@app.cls(image=base_image, gpu="A100", timeout=300, volumes={MODEL_PATH: volume})
class OmniGen2GPU:
    _is_loaded = False
    

    @modal.enter()
    def load_pipeline(self):
        import sys, os
        import torch
        
        # Add OmniGen2 to Python path BEFORE importing
        sys.path.insert(0, "/app/omnigen2")
        
        from accelerate import Accelerator
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
        
        if self._is_loaded:
            return
            
        
        self.model_path = MODEL_PATH
        
        # Log GPU information
        try:
            print(f"[GPU START] CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"[GPU START] GPU Device: {torch.cuda.get_device_name(0)}")
                print(f"[GPU START] CUDA Version: {torch.version.cuda}")
                print(f"[GPU START] Current GPU Memory Usage: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
        except Exception as e:
            print(f"[GPU START] Could not get GPU information: {str(e)}")
        
  
        # Initialize accelerator
        accelerator = Accelerator(mixed_precision="bf16")
        
        # Set weight dtype
        weight_dtype = torch.bfloat16
        
        print(f"[GPU START] Loading OmniGen2 pipeline with dtype: bf16")
        
        # Load pipeline
        pipeline = OmniGen2Pipeline.from_pretrained(
            self.model_path,
            torch_dtype=weight_dtype,
            trust_remote_code=True,
        )
        
        # Load transformer
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            self.model_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )
        
        # Move to device
        pipeline = pipeline.to(accelerator.device)
        print("[GPU START] Pipeline loaded successfully!")
        
        self.pipeline = pipeline
        self.accelerator = accelerator
        self._is_loaded = True

    def preprocess_images(self, input_images: list = None):
        """Preprocess input images following inference.py patterns"""
        if not input_images:
            return None
        
        from PIL import Image, ImageOps
        import io
        
        processed_images = []
        for img_b64 in input_images:
            try:
                # Decode base64
                img_data = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                # Apply EXIF orientation correction like inference.py
                img = ImageOps.exif_transpose(img)
                processed_images.append(img)
            except Exception as e:
                print(f"[GPU TASK] Error processing input image: {e}")
                continue
        
        return processed_images if processed_images else None

    def create_collage(self, images):
        """Create horizontal collage following inference.py patterns"""
        from torchvision.transforms.functional import to_pil_image, to_tensor
        import torch
        
        # Convert PIL images to tensors
        vis_images = [to_tensor(image) * 2 - 1 for image in images]
        
        # Create horizontal collage
        max_height = max(img.shape[-2] for img in vis_images)
        total_width = sum(img.shape[-1] for img in vis_images)
        canvas = torch.zeros((3, max_height, total_width), device=vis_images[0].device)
        
        current_x = 0
        for img in vis_images:
            h, w = img.shape[-2:]
            canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
            current_x += w
        
        return to_pil_image(canvas)

    @modal.method()
    def generate_images(self, prompt: str, negative_prompt: str = None, input_images: list = None, 
                       width: int = 1024, height: int = 1024, num_inference_steps: int = 50,
                       text_guidance_scale: float = 5.0, image_guidance_scale: float = 2.0,
                       cfg_range_start: float = 0.0, cfg_range_end: float = 1.0,
                       num_images_per_prompt: int = 1, seed: int = None, 
                       scheduler: str = "euler", dtype: str = "bf16") -> bytes:
        
        print("[GPU TASK] generate_images started")
        import sys, os, io
        import torch
        sys.path.insert(0, "/app/omnigen2")
        
        # Set default negative prompt
        if negative_prompt is None:
            negative_prompt = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
        
        # Use pre-loaded pipeline and accelerator
        pipeline = self.pipeline
        accelerator = self.accelerator
        
        # Set scheduler if specified
        if scheduler == "dpmsolver++":
            from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
            scheduler_obj = DPMSolverMultistepScheduler(
                algorithm_type="dpmsolver++",
                solver_type="midpoint",
                solver_order=2,
                prediction_type="flow_prediction",
            )
            pipeline.scheduler = scheduler_obj
            print(f"[GPU TASK] Using DPMSolver++ scheduler")
        
        # Preprocess input images
        input_images_processed = self.preprocess_images(input_images)
        
        # Set random seed
        if seed is not None:
            generator = torch.Generator(device=accelerator.device).manual_seed(seed)
        else:
            generator = None
        
        print(f"[GPU TASK] Generating images with prompt: {prompt}")
        print(f"[GPU TASK] Parameters: {width}x{height}, {num_inference_steps} steps, scheduler: {scheduler}")
        
        # Generate images following inference.py run() function
        results = pipeline(
            prompt=prompt,
            input_images=input_images_processed,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            max_sequence_length=1024,
            text_guidance_scale=text_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            cfg_range=(cfg_range_start, cfg_range_end),
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            output_type="pil",
        )
      
        # Always create collage like the original inference.py
        collage = self.create_collage(results.images)
        buffer = io.BytesIO()
        collage.save(buffer, format="PNG")
        return buffer.getvalue()



@app.function(image=base_image)
def validate_image_inputs(prompt: str, input_images: list = None) -> tuple[str, list]:
    print("[CPU VALIDATION] Starting image input validation...")
    
    import filetype
    from PIL import Image
    import io
    
    # Validate prompt
    if not prompt or not prompt.strip():
        raise Exception("400: Prompt cannot be empty")
    
    if len(prompt) > 1000:
        raise Exception("400: Prompt too long (max 1000 characters)")
    
    # Validate input images if provided
    validated_images = []
    if input_images:
        for i, img_b64 in enumerate(input_images):
            try:
                # Validate base64
                img_data = base64.b64decode(img_b64)
                
                # Validate file type
                kind = filetype.guess(img_data)
                if kind is None or not kind.mime.startswith("image/"):
                    raise Exception(f"400: Input image {i+1} is not a valid image")
                
                # Validate image can be opened
                img = Image.open(io.BytesIO(img_data))
                img.verify()
                
                validated_images.append(img_b64)
                
            except Exception as e:
                raise Exception(f"400: Invalid input image {i+1}: {str(e)}")
    
    print("[CPU VALIDATION] Image validation complete")
    return prompt, validated_images



from fastapi import HTTPException, Depends, status, Request,UploadFile
@app.function(image=base_image)
@modal.fastapi_endpoint(method="POST")
def web_endpoint_image(
    prompt: Annotated[str, fastapi.Form(description="Text prompt for image generation")],
    negative_prompt: Annotated[str, fastapi.Form(description="Negative prompt")] = None,
    width: Annotated[int, fastapi.Form(description="Image width")] = 1024,
    height: Annotated[int, fastapi.Form(description="Image height")] = 1024,
    num_inference_steps: Annotated[int, fastapi.Form(description="Number of inference steps")] = 50,
    text_guidance_scale: Annotated[float, fastapi.Form(description="Text guidance scale")] = 5.0,
    image_guidance_scale: Annotated[float, fastapi.Form(description="Image guidance scale")] = 2.0,
    num_images_per_prompt: Annotated[int, fastapi.Form(description="Number of images to generate")] = 1,
    seed: Annotated[int, fastapi.Form(description="Random seed")] = None,
    scheduler: Annotated[str, fastapi.Form(description="Scheduler (euler or dpmsolver++)")] = "euler",
    dtype: Annotated[str, fastapi.Form(description="Data type (fp32, fp16, bf16)")] = "bf16",
    task_type: Annotated[str, fastapi.Form(description="Task type: text2img, editing, in_context")] = "text2img",
    input_images: Annotated[list[UploadFile], fastapi.File(description="Input images (optional)")] = None,
    request: Request = None
) -> Response:
    
    print("[API] Starting image generation request...")
    
    try:
        # Convert input images to base64 if provided
        input_images_b64 = None
        if input_images:
            input_images_b64 = []
            for upload_file in input_images:
                if upload_file.content_type and upload_file.content_type.startswith("image/"):
                    file_content = upload_file.file.read()
                    input_images_b64.append(base64.b64encode(file_content).decode())
        
        # Adjust defaults based on task type
        adjusted_text_guidance_scale = text_guidance_scale
        adjusted_image_guidance_scale = image_guidance_scale
        
        if task_type == "in_context":
            # For in-context generation: preserve more details from input images
            # Recommended range: 2.5-3.0
            if image_guidance_scale == 2.0:  # Only adjust if user didn't specify custom value
                adjusted_image_guidance_scale = 2.8  # Middle of recommended range
            if text_guidance_scale == 5.0:  # Only adjust if user didn't specify custom value
                adjusted_text_guidance_scale = 6.0
            print(f"[API] Task: in_context - Using image_guidance_scale: {adjusted_image_guidance_scale}, text_guidance_scale: {adjusted_text_guidance_scale}")
        elif task_type == "editing":
            # For image editing: allow more modifications while maintaining structure
            # Recommended range: 1.2-2.0
            if image_guidance_scale == 2.0:  # Only adjust if user didn't specify custom value
                adjusted_image_guidance_scale = 1.6  # Middle of recommended range
            print(f"[API] Task: editing - Using image_guidance_scale: {adjusted_image_guidance_scale}")
        elif task_type == "text2img":
            # For text-to-image: no input images, so image_guidance_scale doesn't matter much
            # But we can set a reasonable default for consistency
            if image_guidance_scale == 2.0:  # Only adjust if user didn't specify custom value
                adjusted_image_guidance_scale = 1.0  # Minimal image guidance for pure text2img
            print(f"[API] Task: text2img - Using image_guidance_scale: {adjusted_image_guidance_scale}")
        else:
            print(f"[API] Task: {task_type} - Using user-specified guidance scales")
        
        # Validate inputs
        validated_prompt, validated_images = validate_image_inputs.remote(prompt, input_images_b64)
        
        print("[API] Validation passed. Now invoking GPU task...")
        
        # Generate images
        result_bytes = OmniGen2GPU().generate_images.remote(
            prompt=validated_prompt,
            negative_prompt=negative_prompt,
            input_images=validated_images,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=adjusted_text_guidance_scale,
            image_guidance_scale=adjusted_image_guidance_scale,
            cfg_range_start=0.0,
            cfg_range_end=1.0,
            num_images_per_prompt=num_images_per_prompt,
            seed=seed,
            scheduler=scheduler,
            dtype=dtype
        )
        
        print("[API] GPU result received. Sending response.")
        return Response(content=result_bytes, media_type="image/png")
        
    except Exception as e:
        message = str(e)
        if message.startswith("400:"):
            raise HTTPException(status_code=400, detail=message[4:].strip())
        raise HTTPException(status_code=500, detail="Internal server error")
