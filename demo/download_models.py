"""
Download required models from Hugging Face for offline demo.
Run once before using predict.py:
    python3 download_models.py

Downloads:
  - roberta-base (478MB) → models/roberta-base/
  - whisper-medium (2.8GB) → models/whisper-medium/
"""
import os, sys
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"


def download_model(model_id, local_name):
    """Download a Hugging Face model to models/{local_name}/."""
    target = MODEL_DIR / local_name
    if target.exists() and (target / "config.json").exists():
        print(f"✓ {model_id} already exists at {target}")
        return

    print(f"Downloading {model_id} → {target}...")
    target.mkdir(parents=True, exist_ok=True)

    if "whisper" in model_id:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        processor = WhisperProcessor.from_pretrained(model_id)
        model.save_pretrained(str(target))
        processor.save_pretrained(str(target))
    else:
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model.save_pretrained(str(target))
        tokenizer.save_pretrained(str(target))

    print(f"✓ {model_id} downloaded to {target}")


if __name__ == "__main__":
    print("=" * 50)
    print("Downloading models for AD Detection Demo")
    print("=" * 50)
    print()

    # RoBERTa-base (~478MB)
    download_model("FacebookAI/roberta-base", "roberta-base")

    # Whisper-medium (~2.8GB)
    download_model("openai/whisper-medium", "whisper-medium")

    print()
    print("All models downloaded. You can now run:")
    print("  python3 predict.py example/sample.wav")
