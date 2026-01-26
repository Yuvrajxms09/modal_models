import modal
import os
import uuid
from typing import List, Optional

app = modal.App("higgs-audio")

volume = modal.Volume.from_name("HiggsAudio-assets")

MODEL_PATH = "/models"

higgs_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:25.01-py3",
        secret=modal.Secret.from_name("nvidia-ngc-secret")
    ).pip_install(
        "descript-audio-codec",
        "torch",
        "transformers>=4.45.1,<4.47.0",
        "librosa",
        "dacite",
        "boto3==1.35.36",
        "s3fs",
        "torchvision",
        "torchaudio",
        "json_repair",
        "pandas",
        "pydantic",
        "vector_quantize_pytorch",
        "loguru",
        "pydub",
        "ruff==0.12.2",
        "omegaconf",
        "click",
        "langid",
        "jieba",
        "accelerate>=0.26.0",
    )
    .add_local_dir("higgs-audio", remote_path="/app/higgs-audio", copy=True, ignore=[".git", "__pycache__", "*.pyc"])
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
    )
)


@app.cls(image=higgs_image, gpu="A100", scaledown_window=180, timeout=2400, volumes={MODEL_PATH: volume}, secrets=[modal.Secret.from_name("aws-s3-secrets")])
@modal.concurrent(max_inputs=3, target_inputs=2)
class HiggsAudioGPU:
    _is_loaded = False

    @modal.enter()
    def load_pipeline(self):
        import sys
        import os
        
        sys.path.insert(0, "/app/higgs-audio")

        if self._is_loaded:
            return
        
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        from boson_multimodal.data_types import ChatMLSample, AudioContent, Message
        
        self.ChatMLSample = ChatMLSample
        self.AudioContent = AudioContent
        self.Message = Message
        
        self.engine = HiggsAudioServeEngine(
            model_name_or_path="/models/higgs-audio-v2-generation-3B-base",
            audio_tokenizer_name_or_path="/models/higgs-audio-v2-tokenizer",
            device="cuda",
        )
        
        self._is_loaded = True

    def upload_to_s3(self, audio_bytes: bytes, audio_index: int = 0) -> str:
        import boto3
        import io
        
        s3 = boto3.client("s3")
        bucket_name = "xxxxxxxx"
        s3_key = f"higgs-audio/higgs_audio_{uuid.uuid4().hex[:12]}_{audio_index}.wav"
        
        s3.upload_fileobj(
            io.BytesIO(audio_bytes),
            bucket_name,
            s3_key,
            ExtraArgs={'ContentType': 'audio/wav'}
        )
        
        return f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"

    def normalize_chinese_punctuation(self, text):
        chinese_to_english_punct = {
            "，": ", ", "。": ".", "：": ":", "；": ";", "？": "?", "！": "!",
            "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
            """: '"', """: '"', "'": "'", "'": "'", "、": ",", "—": "-",
            "…": "...", "·": ".", """: '"', """: '"', """: '"', """: '"',
        }
        
        for zh_punct, en_punct in chinese_to_english_punct.items():
            text = text.replace(zh_punct, en_punct)
        return text

    def normalize_text(self, transcript: str):
        transcript = self.normalize_chinese_punctuation(transcript)
        transcript = transcript.replace("(", " ")
        transcript = transcript.replace(")", " ")
        transcript = transcript.replace("°F", " degrees Fahrenheit")
        transcript = transcript.replace("°C", " degrees Celsius")

        for tag, replacement in [
            ("[laugh]", "<SE>[Laughter]</SE>"),
            ("[humming start]", "<SE>[Humming]</SE>"),
            ("[humming end]", "<SE_e>[Humming]</SE_e>"),
            ("[music start]", "<SE_s>[Music]</SE_s>"),
            ("[music end]", "<SE_e>[Music]</SE_e>"),
            ("[music]", "<SE>[Music]</SE>"),
            ("[sing start]", "<SE_s>[Singing]</SE_s>"),
            ("[sing end]", "<SE_e>[Singing]</SE_e>"),
            ("[applause]", "<SE>[Applause]</SE>"),
            ("[cheering]", "<SE>[Cheering]</SE>"),
            ("[cough]", "<SE>[Cough]</SE>"),
        ]:
            transcript = transcript.replace(tag, replacement)

        lines = transcript.split("\n")
        transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
        transcript = transcript.strip()

        if not any([transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
            transcript += "."

        return transcript

    def prepare_chatml_sample(self, text: str, system_prompt: str = None, reference_audio_base64: str = None, reference_text: str = None):
        messages = []

        if system_prompt and len(system_prompt.strip()) > 0:
            messages.append(self.Message(role="system", content=system_prompt))

        if reference_audio_base64:
            messages.append(self.Message(role="user", content=reference_text or ""))
            messages.append(self.Message(role="assistant", content=[self.AudioContent(raw_audio=reference_audio_base64, audio_url="")]))

        messages.append(self.Message(role="user", content=self.normalize_text(text)))
        return self.ChatMLSample(messages=messages)

    @modal.method()
    def generate_audio_direct(self,
                             text: str,
                             system_prompt: str = None,
                             reference_audio_base64: str = None,
                             reference_text: str = None,
                             max_completion_tokens: int = 1024,
                             temperature: float = 1.0,
                             top_p: float = 0.95,
                             top_k: int = 50,
                             ras_win_len: int = 7,
                             ras_win_max_num_repeat: int = 2,
                             stop_strings: List[str] = None
                             ) -> bytes:
        import tempfile
        import torchaudio
        import torch
        
        if system_prompt is None:
            system_prompt = (
                "Generate audio following instruction.\n\n"
                "<|scene_desc_start|>\n"
                "Audio is recorded from a quiet room.\n"
                "<|scene_desc_end|>"
            )
        
        if stop_strings is None:
            stop_strings = ["<|end_of_text|>", "<|eot_id|>"]
        
        chatml_sample = self.prepare_chatml_sample(
            text=text,
            system_prompt=system_prompt,
            reference_audio_base64=reference_audio_base64,
            reference_text=reference_text
        )

        response = self.engine.generate(
            chat_ml_sample=chatml_sample,
            max_new_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p,
            stop_strings=stop_strings,
            ras_win_len=ras_win_len if ras_win_len > 0 else None,
            ras_win_max_num_repeat=ras_win_max_num_repeat,
        )

        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()
        
        torchaudio.save(temp_audio_path, torch.from_numpy(response.audio)[None, :], response.sampling_rate)
        
        with open(temp_audio_path, "rb") as f:
            audio_bytes = f.read()
        
        os.remove(temp_audio_path)
        return audio_bytes

    @modal.method()
    def generate_audio_and_upload(self, 
                                 text: str,
                                 system_prompt: str = None,
                                 reference_audio_base64: str = None,
                                 reference_text: str = None,
                                 max_completion_tokens: int = 1024,
                                 temperature: float = 1.0,
                                 top_p: float = 0.95,
                                 top_k: int = 50,
                                 ras_win_len: int = 7,
                                 ras_win_max_num_repeat: int = 2,
                                 stop_strings: List[str] = None
                                 ) -> str:
        import tempfile
        import torchaudio
        import torch
        
        if system_prompt is None:
            system_prompt = (
                "Generate audio following instruction.\n\n"
                "<|scene_desc_start|>\n"
                "Audio is recorded from a quiet room.\n"
                "<|scene_desc_end|>"
            )
        
        if stop_strings is None:
            stop_strings = ["<|end_of_text|>", "<|eot_id|>"]
        
        chatml_sample = self.prepare_chatml_sample(
            text=text,
            system_prompt=system_prompt,
            reference_audio_base64=reference_audio_base64,
            reference_text=reference_text
        )

        response = self.engine.generate(
            chat_ml_sample=chatml_sample,
            max_new_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p,
            stop_strings=stop_strings,
            ras_win_len=ras_win_len if ras_win_len > 0 else None,
            ras_win_max_num_repeat=ras_win_max_num_repeat,
        )

        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()
        
        torchaudio.save(temp_audio_path, torch.from_numpy(response.audio)[None, :], response.sampling_rate)
        
        with open(temp_audio_path, "rb") as f:
            audio_bytes = f.read()
        
        os.remove(temp_audio_path)
        return self.upload_to_s3(audio_bytes, 0)

@app.function(image=web_image, timeout=2400)
@modal.asgi_app(label="higgs-audio-web-endpoint")
def fastapi_app():
    from fastapi import FastAPI, Request, Form
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel, Field, ConfigDict
    
    class HiggsAudioRequestAPI(BaseModel):
        model_config = ConfigDict(protected_namespaces=())
        text: str
        system_prompt: Optional[str] = None
        reference_audio_base64: Optional[str] = None
        reference_text: Optional[str] = None
        max_completion_tokens: int = 1024
        temperature: float = 1.0
        top_p: float = 0.95
        top_k: int = 50
        ras_win_len: int = 7
        ras_win_max_num_repeat: int = 2

    class HiggsAudioResponseAPI(BaseModel):
        url: str
        status: str = "success"
        message: str = "Audio generated successfully"

    web_app = FastAPI(title="Higgs Audio Text-to-Speech API", version="1.0.0")

    @web_app.post("/generate")
    async def generate_audio_endpoint(
        text: str,
        system_prompt: Optional[str] = None,
        reference_audio_base64: Optional[str] = None,
        reference_text: Optional[str] = None,
        max_completion_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        ras_win_len: int = 7,
        ras_win_max_num_repeat: int = 2
    ):
        try:
            result_bytes = HiggsAudioGPU().generate_audio_direct.remote(
                text=text,
                system_prompt=system_prompt,
                reference_audio_base64=reference_audio_base64,
                reference_text=reference_text,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat
            )
            return Response(content=result_bytes, media_type="audio/wav")
        except Exception as e:
            error_msg = str(e)
            status_code = 400 if error_msg.startswith("400:") else 500
            message = error_msg[4:].strip() if error_msg.startswith(("400:", "500:")) else "Internal server error occurred. Please try again."
            
            return JSONResponse(
                status_code=status_code,
                content={"status": "error", "message": message}
            )

    @web_app.post("/generate-async", response_model=HiggsAudioResponseAPI)
    async def generate_audio_async_endpoint(request_data: HiggsAudioRequestAPI):
        try:
            public_url = HiggsAudioGPU().generate_audio_and_upload.remote(
                text=request_data.text,
                system_prompt=request_data.system_prompt,
                reference_audio_base64=request_data.reference_audio_base64,
                reference_text=request_data.reference_text,
                max_completion_tokens=request_data.max_completion_tokens,
                temperature=request_data.temperature,
                top_p=request_data.top_p,
                top_k=request_data.top_k,
                ras_win_len=request_data.ras_win_len,
                ras_win_max_num_repeat=request_data.ras_win_max_num_repeat
            )
            return HiggsAudioResponseAPI(url=public_url)
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
        return {"status": "healthy", "service": "higgs-audio"}

    return web_app
