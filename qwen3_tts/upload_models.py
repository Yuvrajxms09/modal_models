import modal
import os

app = modal.App(name="model-uploader")

# Define the volume and mount path at the top level
MODEL_PATH = "/models"
volume = modal.Volume.from_name("qwen3-TTS-assets", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .apt_install("git-lfs")
    .run_commands("git lfs install")
    .pip_install("huggingface_hub", "torch")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
    volumes={MODEL_PATH: volume}  # Mount the volume here
)
def upload_models_to_volume(models: list):
    from huggingface_hub import snapshot_download
    
    hf_token = os.environ.get("HF_TOKEN")
    
    for model_config in models:
        model_id = model_config["model_id"]
        # Use relative path within the mounted volume
        local_dir = model_config.get("local_dir", "").lstrip("/")
        target_path = os.path.join(MODEL_PATH, local_dir)
        
        if not model_id:
            print(f"Skipping empty model_id")
            continue
        
        print(f"Downloading {model_id} to {target_path}...")
        
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=target_path,
                token=hf_token,
                ignore_patterns=["*.git*", "*.gitattributes", ".git*"]
            )
            
            # Modal Volumes require an explicit commit to persist changes 
            # made during the function execution
            volume.commit()
            print(f"Successfully downloaded and committed {model_id}")
            
            # Print file info
            if os.path.exists(target_path):
                print(f"Files in {target_path}:")
                total_size = 0
                for root, dirs, files in os.walk(target_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path) / (1024 * 1024)
                        total_size += file_size
                        rel_path = os.path.relpath(file_path, target_path)
                        print(f"   - {rel_path} ({file_size:.2f} MB)")
                print(f"Total size: {total_size:.2f} MB")
                
        except Exception as e:
            print(f"Error downloading {model_id}: {e}")
            raise


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret")], timeout=1800)
def download_models():
    models_config = [
        {
            "model_id": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            "local_dir": "Qwen3-TTS-Tokenizer-12Hz"
        },
        {
            "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "local_dir": "Qwen3-TTS-12Hz-1.7B-CustomVoice"
        },
        {
            "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "local_dir": "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        },
        {
            "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "local_dir": "Qwen3-TTS-12Hz-1.7B-Base"
        },
        {
            "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "local_dir": "Qwen3-TTS-12Hz-0.6B-CustomVoice"
        },
        {
            "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "local_dir": "Qwen3-TTS-12Hz-0.6B-Base"
        }
    ]
    
    upload_models_to_volume.remote(models_config)


@app.local_entrypoint()
def main():
    download_models.remote()
