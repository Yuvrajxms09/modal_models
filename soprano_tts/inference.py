import os
import uuid
import tempfile
import modal

app = modal.App("soprano_tts")

volume = modal.Volume.from_name("TTS-assets", create_if_missing=False)

MODEL_PATH = "/models"


soprano_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install("git")
    .uv_pip_install(
        # Core Soprano TTS dependencies
        "torch>=2.1.0",
        "transformers>=4.51.0",
        "accelerate",
        "numpy",
        "scipy",
        "huggingface_hub",
        "unidecode",
        "inflect",
        "sounddevice",
        # LMDeploy backend for faster inference
        "lmdeploy",
        # Additional packages for Modal
        "fastapi",
        "uvicorn",
        "gradio",
        extra_options="--index-strategy unsafe-best-match",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    # Clone Soprano repository
    .run_commands("git clone --single-branch --branch main https://github.com/ekwek1/soprano /soprano-repo")
    # Install Soprano from source
    .run_commands("cd /soprano-repo && pip install -e .[lmdeploy]")
)


# Web image for API server
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "pydantic==2.5.0",
        "python-multipart==0.0.6",
    )
)


@app.cls(image=soprano_image, scaledown_window=180, timeout=600, volumes={MODEL_PATH: volume})
@modal.concurrent(max_inputs=4, target_inputs=3)
class SopranoTTSWorker:
    _is_loaded = False

    @modal.enter()
    def load_pipeline(self):
        import time
        import torch
        import sys
        from soprano import SopranoTTS

        if self._is_loaded:
            return

        print("[CPU START] Loading Soprano TTS pipeline...")
        start_time = time.time()

        sys.path.insert(0, "/soprano-repo")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        self.pipeline = SopranoTTS(
            backend='transformers',
            device='cpu',
            cache_size_mb=100,
            decoder_batch_size=1,
            model_path=None
        )

        load_time = time.time() - start_time
        print(f"[CPU START] Pipeline loaded in {load_time:.1f}s!")
        self._is_loaded = True

    def save_to_volume(self, audio_path: str) -> str:
        import shutil
        
        output_dir = os.path.join(MODEL_PATH, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"soprano_audio_{uuid.uuid4().hex[:12]}.wav"
        volume_path = os.path.join(output_dir, filename)
        shutil.copy2(audio_path, volume_path)
        
        return volume_path

    def process_single_generation(self, text: str, temperature: float = 0.3, top_p: float = 0.95, repetition_penalty: float = 1.2) -> str:
        import time
        import torch
        import numpy as np
        from scipy.io.wavfile import write

        print(f"[CPU TASK] Generating audio for text: {text[:100]}...")
        start_time = time.time()

        with torch.no_grad():
            audio_tensor = self.pipeline.infer(
                text=text,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        print(f"[CPU TASK] Audio generation completed in {time.time() - start_time:.1f}s")

        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, "generated_audio.wav")

        audio_numpy = audio_tensor.cpu().numpy()
        audio_int16 = (np.clip(audio_numpy, -1.0, 1.0) * 32767).astype(np.int16)
        write(temp_audio_path, 32000, audio_int16)

        return temp_audio_path

    @modal.method()
    def generate_audio_and_save(self, text: str, temperature: float = 0.3, top_p: float = 0.95, repetition_penalty: float = 1.2) -> tuple:
        temp_audio_path = self.process_single_generation(text, temperature, top_p, repetition_penalty)
        
        with open(temp_audio_path, 'rb') as f:
            audio_data = f.read()
        
        volume_path = self.save_to_volume(temp_audio_path)
        
        os.remove(temp_audio_path)
        os.rmdir(os.path.dirname(temp_audio_path))
        
        return volume_path, audio_data


@app.function(image=web_image)
@modal.asgi_app(label="soprano-web-endpoint")
def fastapi_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel, Field, ConfigDict
    
    class SpeechRequest(BaseModel):
        model_config = ConfigDict(protected_namespaces=())
        input: str = Field(..., max_length=5000)
        temperature: float = Field(default=0.3, ge=0.0, le=2.0)
        top_p: float = Field(default=0.95, ge=0.0, le=1.0)
        repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)

    web_app = FastAPI(title="Soprano TTS API", version="1.0.0")

    @web_app.post("/v1/audio/speech")
    async def generate_speech_endpoint(request: SpeechRequest):
        try:
            volume_path, audio_data = SopranoTTSWorker().generate_audio_and_save.remote(
                text=request.input,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty
            )
            
            return Response(
                content=audio_data,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": 'attachment; filename="speech.wav"',
                    "X-Modal-Volume-Path": volume_path
                }
            )
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
        return {"status": "healthy", "service": "soprano-tts"}

    return web_app
