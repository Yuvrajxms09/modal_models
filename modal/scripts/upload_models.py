import modal

app = modal.App(name="model-uploader")

MODEL_PATH = "/models"

image = (
    modal.Image.debian_slim()
    .apt_install("git-lfs")
    .run_commands("git lfs install")
    .pip_install("huggingface_hub", "torch")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800
)
def upload_models_to_volume(volume_name: str, models: list):
    from huggingface_hub import snapshot_download
    import os
    
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    
    with volume.mount(MODEL_PATH):
        hf_token = os.environ.get("HF_TOKEN")
        
        for model_config in models:
            model_id = model_config["model_id"]
            local_dir = model_config.get("local_dir", MODEL_PATH)
            target_path = os.path.join(MODEL_PATH, local_dir.lstrip("/"))
            
            if not model_id:
                print(f"  Skipping empty model_id")
                continue
            
            print(f" Downloading {model_id}...")
            
            try:
                snapshot_download(
                    repo_id=model_id,
                    local_dir=target_path,
                    token=hf_token,
                    ignore_patterns=["*.git*", "*.gitattributes", ".git*"]
                )
                print(f"Model '{model_id}' downloaded to {target_path}")
                
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
                print(f"Error downloading {model_id}: {str(e)}")
                raise
        
        volume.commit()
        print(f" Volume '{volume_name}' committed successfully!")


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret")], timeout=1800)
def download_models():
    from huggingface_hub import snapshot_download
    import os
    
    models_config = [
        {
            "model_id": "",  # Add your HuggingFace model ID here
            "local_dir": "/models"  # Subdirectory in volume
        }
    ]
    
    volume_name = "volume-name"  # Update with your volume name
    
    if not models_config or not models_config[0].get("model_id"):
        print("No models configured. Please update the models_config list.")
        return
    
    upload_models_to_volume.remote(volume_name, models_config)


if __name__ == "__main__":
    with app.run():
        download_models.remote()
