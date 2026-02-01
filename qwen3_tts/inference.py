import os
import uuid
import tempfile
import modal
from typing import Optional

app = modal.App("qwen3_tts")

volume = modal.Volume.from_name("qwen3-TTS-assets", create_if_missing=False)

MODEL_PATH = "/models"

qwen3_tts_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        "transformers==4.57.3",
        "accelerate==1.12.0",
        "librosa",
        "soundfile",
        "sox",
        "onnxruntime",
        "einops",
        "torch==2.8.0",
        "torchaudio",
        "numpy",
        "scipy",
        "huggingface_hub",
        "gradio",
        extra_options="--index-strategy unsafe-best-match",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands("pip install wheel setuptools")
    .run_commands("pip install -U flash-attn --no-build-isolation || echo 'FlashAttention installation failed, continuing without it'")
    .run_commands("pip install -U qwen-tts")
)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "pydantic==2.5.0",
        "python-multipart==0.0.6",
    )
)


@app.cls(image=qwen3_tts_image, gpu="T4", scaledown_window=180, timeout=600, volumes={MODEL_PATH: volume})
@modal.concurrent(max_inputs=4, target_inputs=3)
class Qwen3TTSWorker:
    _is_loaded = False
    _model_type = None

    @modal.enter()
    def load_pipeline(self):
        import time
        import torch
        from qwen_tts import Qwen3TTSModel

        if self._is_loaded:
            return

        print("[GPU START] Loading Qwen3-TTS pipeline...")
        start_time = time.time()

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        model_name = os.environ.get("QWEN3_TTS_MODEL", "Qwen3-TTS-12Hz-1.7B-CustomVoice")
        
        # Load model from volume
        local_model_path = os.path.join(MODEL_PATH, model_name)
        print(f"[GPU START] Loading model from volume: {local_model_path}")
        
        try:
            self.model = Qwen3TTSModel.from_pretrained(
                local_model_path,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            
            if "CustomVoice" in model_name:
                self._model_type = "custom_voice"
            elif "VoiceDesign" in model_name:
                self._model_type = "voice_design"
            elif "Base" in model_name:
                self._model_type = "voice_clone"
            else:
                self._model_type = "custom_voice"
            
            load_time = time.time() - start_time
            print(f"[GPU START] Pipeline loaded in {load_time:.1f}s! Model type: {self._model_type}")
            self._is_loaded = True
        except Exception as e:
            print(f"[GPU START] Error loading model: {e}")
            raise

    def save_to_volume(self, audio_path: str) -> str:
        import shutil
        
        output_dir = os.path.join(MODEL_PATH, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"qwen3_tts_audio_{uuid.uuid4().hex[:12]}.wav"
        volume_path = os.path.join(output_dir, filename)
        shutil.copy2(audio_path, volume_path)
        
        return volume_path

    def process_custom_voice(self, text: str, language: str = "Auto", speaker: str = "Vivian", instruct: Optional[str] = None, max_new_tokens: int = 2048, top_p: Optional[float] = None, temperature: Optional[float] = None, repetition_penalty: Optional[float] = None) -> str:
        import time
        import soundfile as sf

        if not text or not text.strip():
            raise ValueError("Text is required for custom voice generation")

        # Normalize speaker name (handle case and spaces)
        speaker_normalized = speaker.strip().lower().replace(" ", "_")

        print(f"[GPU TASK] Generating custom voice audio for text: {text[:100]}...")
        start_time = time.time()

        gen_kwargs = {
            "non_streaming_mode": True,
            "max_new_tokens": max_new_tokens,
        }
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        
        wavs, sr = self.model.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker_normalized,
            instruct=instruct.strip() if instruct else None,
            **gen_kwargs,
        )

        print(f"[GPU TASK] Audio generation completed in {time.time() - start_time:.1f}s")

        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, "generated_audio.wav")
        sf.write(temp_audio_path, wavs[0], sr)

        return temp_audio_path

    def process_voice_design(self, text: str, language: str = "Auto", instruct: str = "", max_new_tokens: int = 2048, top_p: Optional[float] = None, temperature: Optional[float] = None, repetition_penalty: Optional[float] = None) -> str:
        import time
        import soundfile as sf

        if not text or not text.strip():
            raise ValueError("Text is required for voice design generation")
        if not instruct or not instruct.strip():
            raise ValueError("Voice description (instruct) is required for voice design")

        print(f"[GPU TASK] Generating voice design audio for text: {text[:100]}...")
        start_time = time.time()

        gen_kwargs = {
            "non_streaming_mode": True,
            "max_new_tokens": max_new_tokens,
        }
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        
        wavs, sr = self.model.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=instruct.strip(),
            **gen_kwargs,
        )

        print(f"[GPU TASK] Audio generation completed in {time.time() - start_time:.1f}s")

        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, "generated_audio.wav")
        sf.write(temp_audio_path, wavs[0], sr)

        return temp_audio_path

    def process_voice_clone(self, text: str, language: str = "Auto", ref_audio: Optional[str] = None, ref_text: Optional[str] = None, x_vector_only_mode: bool = False, max_new_tokens: int = 2048, top_p: Optional[float] = None, temperature: Optional[float] = None, repetition_penalty: Optional[float] = None) -> str:
        import time
        import soundfile as sf

        if not text or not text.strip():
            raise ValueError("Target text is required for voice clone generation")
        if not ref_audio:
            raise ValueError("Reference audio is required for voice clone model")
        if not x_vector_only_mode and (not ref_text or not ref_text.strip()):
            raise ValueError("Reference text is required when x_vector_only_mode is False")

        print(f"[GPU TASK] Generating voice clone audio for text: {text[:100]}...")
        start_time = time.time()

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
        }
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        
        wavs, sr = self.model.generate_voice_clone(
            text=text.strip(),
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=x_vector_only_mode,
            **gen_kwargs,
        )

        print(f"[GPU TASK] Audio generation completed in {time.time() - start_time:.1f}s")

        temp_dir = tempfile.mkdtemp()
        temp_audio_path = os.path.join(temp_dir, "generated_audio.wav")
        sf.write(temp_audio_path, wavs[0], sr)

        return temp_audio_path

    @modal.method()
    def generate_custom_voice_and_save(self, text: str, language: str = "Auto", speaker: str = "Vivian", instruct: Optional[str] = None, max_new_tokens: int = 2048, top_p: Optional[float] = None, temperature: Optional[float] = None, repetition_penalty: Optional[float] = None) -> tuple:
        temp_audio_path = self.process_custom_voice(text, language, speaker, instruct, max_new_tokens, top_p, temperature, repetition_penalty)
        
        with open(temp_audio_path, 'rb') as f:
            audio_data = f.read()
        
        volume_path = self.save_to_volume(temp_audio_path)
        
        os.remove(temp_audio_path)
        os.rmdir(os.path.dirname(temp_audio_path))
        
        return volume_path, audio_data

    @modal.method()
    def generate_voice_design_and_save(self, text: str, language: str = "Auto", instruct: str = "", max_new_tokens: int = 2048, top_p: Optional[float] = None, temperature: Optional[float] = None, repetition_penalty: Optional[float] = None) -> tuple:
        temp_audio_path = self.process_voice_design(text, language, instruct, max_new_tokens, top_p, temperature, repetition_penalty)
        
        with open(temp_audio_path, 'rb') as f:
            audio_data = f.read()
        
        volume_path = self.save_to_volume(temp_audio_path)
        
        os.remove(temp_audio_path)
        os.rmdir(os.path.dirname(temp_audio_path))
        
        return volume_path, audio_data

    @modal.method()
    def generate_voice_clone_and_save(self, text: str, language: str = "Auto", ref_audio: Optional[str] = None, ref_text: Optional[str] = None, x_vector_only_mode: bool = False, max_new_tokens: int = 2048, top_p: Optional[float] = None, temperature: Optional[float] = None, repetition_penalty: Optional[float] = None) -> tuple:
        temp_audio_path = self.process_voice_clone(text, language, ref_audio, ref_text, x_vector_only_mode, max_new_tokens, top_p, temperature, repetition_penalty)
        
        with open(temp_audio_path, 'rb') as f:
            audio_data = f.read()
        
        volume_path = self.save_to_volume(temp_audio_path)
        
        os.remove(temp_audio_path)
        os.rmdir(os.path.dirname(temp_audio_path))
        
        return volume_path, audio_data

    @modal.method()
    def get_model_info(self) -> dict:
        return {
            "model_type": self._model_type,
            "is_loaded": self._is_loaded,
            "supported_speakers": self.model.get_supported_speakers() if hasattr(self.model, 'get_supported_speakers') else [],
            "supported_languages": self.model.get_supported_languages() if hasattr(self.model, 'get_supported_languages') else [],
        }


@app.function(image=web_image)
@modal.asgi_app(label="qwen3-tts-web-endpoint")
def fastapi_app():
    from fastapi import FastAPI, Form, UploadFile, File
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel, Field, ConfigDict
    from typing import Optional
    
    class CustomVoiceRequest(BaseModel):
        model_config = ConfigDict(protected_namespaces=())
        input: str = Field(..., max_length=5000, description="Text to synthesize")
        language: str = Field(default="Auto", description="Language (or 'Auto' for auto-detection)")
        speaker: str = Field(default="Vivian", description="Speaker name (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee)")
        instruct: Optional[str] = Field(default=None, description="Optional style instruction (e.g., '用特别愤怒的语气说')")
        max_new_tokens: Optional[int] = Field(default=2048, ge=1, le=4096, description="Maximum number of tokens to generate")
        top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
        temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature")
        repetition_penalty: Optional[float] = Field(default=None, ge=1.0, le=2.0, description="Repetition penalty")

    class VoiceDesignRequest(BaseModel):
        model_config = ConfigDict(protected_namespaces=())
        input: str = Field(..., max_length=5000, description="Text to synthesize")
        language: str = Field(default="Auto", description="Language (or 'Auto' for auto-detection)")
        instruct: str = Field(..., description="Voice design instruction (required, describes voice characteristics)")
        max_new_tokens: Optional[int] = Field(default=2048, ge=1, le=4096, description="Maximum number of tokens to generate")
        top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
        temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature")
        repetition_penalty: Optional[float] = Field(default=None, ge=1.0, le=2.0, description="Repetition penalty")

    class VoiceCloneRequest(BaseModel):
        model_config = ConfigDict(protected_namespaces=())
        input: str = Field(..., max_length=5000, description="Target text to synthesize with cloned voice")
        language: str = Field(default="Auto", description="Language (or 'Auto' for auto-detection)")
        ref_audio: str = Field(..., description="Reference audio: URL, local file path, base64 string, or (numpy_array, sample_rate) tuple")
        ref_text: Optional[str] = Field(default=None, description="Transcript of reference audio (required unless x_vector_only_mode=True)")
        x_vector_only_mode: bool = Field(default=False, description="Use only speaker embedding (ref_text not required, but lower quality)")
        max_new_tokens: Optional[int] = Field(default=2048, ge=1, le=4096, description="Maximum number of tokens to generate")
        top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
        temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature")
        repetition_penalty: Optional[float] = Field(default=None, ge=1.0, le=2.0, description="Repetition penalty")

    web_app = FastAPI(title="Qwen3-TTS API", version="1.0.0")

    @web_app.post("/v1/audio/speech/custom-voice")
    async def generate_custom_voice_endpoint(request: CustomVoiceRequest):
        try:
            volume_path, audio_data = Qwen3TTSWorker().generate_custom_voice_and_save.remote(
                text=request.input,
                language=request.language,
                speaker=request.speaker,
                instruct=request.instruct,
                max_new_tokens=request.max_new_tokens,
                top_p=request.top_p,
                temperature=request.temperature,
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
            message = error_msg[4:].strip() if error_msg.startswith(("400:", "500:")) else f"Error: {error_msg}"
            
            return JSONResponse(
                status_code=status_code,
                content={"status": "error", "message": message}
            )

    @web_app.post("/v1/audio/speech/voice-design")
    async def generate_voice_design_endpoint(request: VoiceDesignRequest):
        try:
            volume_path, audio_data = Qwen3TTSWorker().generate_voice_design_and_save.remote(
                text=request.input,
                language=request.language,
                instruct=request.instruct,
                max_new_tokens=request.max_new_tokens,
                top_p=request.top_p,
                temperature=request.temperature,
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
            message = error_msg[4:].strip() if error_msg.startswith(("400:", "500:")) else f"Error: {error_msg}"
            
            return JSONResponse(
                status_code=status_code,
                content={"status": "error", "message": message}
            )

    @web_app.post("/v1/audio/speech/voice-clone")
    async def generate_voice_clone_endpoint(request: VoiceCloneRequest):
        try:
            volume_path, audio_data = Qwen3TTSWorker().generate_voice_clone_and_save.remote(
                text=request.input,
                language=request.language,
                ref_audio=request.ref_audio,
                ref_text=request.ref_text,
                x_vector_only_mode=request.x_vector_only_mode,
                max_new_tokens=request.max_new_tokens,
                top_p=request.top_p,
                temperature=request.temperature,
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
            message = error_msg[4:].strip() if error_msg.startswith(("400:", "500:")) else f"Error: {error_msg}"
            
            return JSONResponse(
                status_code=status_code,
                content={"status": "error", "message": message}
            )

    @web_app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "qwen3-tts"}

    @web_app.get("/model-info")
    async def get_model_info():
        try:
            model_info = Qwen3TTSWorker().get_model_info.remote()
            return model_info
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "note": "Set QWEN3_TTS_MODEL environment variable to specify model (e.g., Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)"
            }

    @web_app.get("/supported-speakers")
    async def get_supported_speakers():
        return {
            "speakers": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"],
            "descriptions": {
                "Vivian": "Bright, slightly edgy young female voice (Chinese)",
                "Serena": "Warm, gentle young female voice (Chinese)",
                "Uncle_Fu": "Seasoned male voice with a low, mellow timbre (Chinese)",
                "Dylan": "Youthful Beijing male voice with a clear, natural timbre (Chinese Beijing Dialect)",
                "Eric": "Lively Chengdu male voice with a slightly husky brightness (Chinese Sichuan Dialect)",
                "Ryan": "Dynamic male voice with strong rhythmic drive (English)",
                "Aiden": "Sunny American male voice with a clear midrange (English)",
                "Ono_Anna": "Playful Japanese female voice with a light, nimble timbre (Japanese)",
                "Sohee": "Warm Korean female voice with rich emotion (Korean)"
            }
        }

    @web_app.get("/supported-languages")
    async def get_supported_languages():
        return {
            "languages": ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian", "Auto"]
        }

    return web_app
