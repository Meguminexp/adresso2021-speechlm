"""
AD Detection Demo — Predict AD probability from a WAV file.

Pipeline: Whisper ASR → RoBERTa CLS + Pause features → Logistic Regression.
"""
import numpy as np, pickle, json, warnings, sys, os
from pathlib import Path
import soundfile as sf
import torch

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

MODEL_DIR = Path(__file__).parent / "models"

# Lazy-loaded globals
_roberta = None
_tokenizer = None
_fold_clfs = None      # list of 5 LR models
_fold_scalers = None   # list of 5 StandardScalers
_libri_ref = None


def _load_models():
    global _roberta, _tokenizer, _fold_clfs, _fold_scalers, _libri_ref
    if _fold_clfs is not None:
        return

    from transformers import AutoModel, AutoTokenizer

    print("Loading RoBERTa...")
    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / "roberta-base"), local_files_only=True)
    _roberta = AutoModel.from_pretrained(str(MODEL_DIR / "roberta-base"), local_files_only=True)

    print("Loading 5-fold IW ensemble...")
    with open(MODEL_DIR / "classifier" / "iw_fold_models.pkl", "rb") as f:
        _fold_clfs = pickle.load(f)
    with open(MODEL_DIR / "classifier" / "iw_fold_scalers.pkl", "rb") as f:
        _fold_scalers = pickle.load(f)
    with open(MODEL_DIR / "classifier" / "librispeech_ref.json") as f:
        _libri_ref = json.load(f)

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _roberta = _roberta.to(_device)
    print(f"  {len(_fold_clfs)} folds loaded on {_device}")


def _extract_roberta_cls(text):
    """Run RoBERTa on text, return CLS token vector (768d)."""
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = _tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(_device)

    with torch.no_grad():
        rob_out = _roberta(**enc).last_hidden_state
    cls_vec = rob_out[0, 0].cpu().numpy()

    return cls_vec.reshape(1, -1).astype(np.float64)


def extract_pause_features(whisper_segments, libri_ref):
    """Extract 60 pause features (z-score vs LibriSpeech reference)."""
    words = [s for s in whisper_segments if not s.get("is_pause")]
    pauses = [s for s in whisper_segments if s.get("is_pause")]
    disfluencies = [s for s in whisper_segments if s.get("is_disfluent")]

    if len(words) < 3:
        return np.zeros(60, dtype=np.float32)

    feats = []
    ref_mean_dur = libri_ref.get("word_duration_mean", 0.3)
    ref_std_dur = libri_ref.get("word_duration_std", 0.18)
    ref_mean_pause = libri_ref.get("pause_mean", 0.35)
    ref_std_pause = libri_ref.get("pause_std", 0.25)
    ref_wps_mean = libri_ref.get("words_per_sec_mean", 2.8)
    ref_wps_std = libri_ref.get("words_per_sec_std", 0.5)
    ref_pause_rate_mean = libri_ref.get("pause_rate_mean", 8.2)
    ref_pause_rate_std = libri_ref.get("pause_rate_std", 3.5)

    durations = np.array([s["end_sec"] - s["start_sec"] for s in words])

    # Word duration percentiles (10)
    for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        feats.append((np.percentile(durations, p) - ref_mean_dur) / (ref_std_dur + 0.01))

    # Duration CV (1)
    feats.append(durations.std() / (durations.mean() + 0.01))

    # Pause duration percentiles (10)
    pause_durs = np.array([s["interval_sec"] for s in pauses]) if pauses else np.zeros(1)
    for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        v = np.percentile(pause_durs, p) if len(pauses) > 0 else 0
        feats.append((v - ref_mean_pause) / (ref_std_pause + 0.01))

    # Pause CV (1)
    feats.append(pause_durs.std() / (pause_durs.mean() + 0.01) if len(pauses) > 0 and pause_durs.mean() > 0 else 0)

    # Position-dependent (4)
    n = len(durations)
    q1 = durations[:max(1, n // 4)].mean()
    q4 = durations[3 * n // 4:].mean()
    feats.append((q1 - ref_mean_dur) / (ref_std_dur + 0.01))
    feats.append((q4 - ref_mean_dur) / (ref_std_dur + 0.01))
    feats.append((durations[n // 4:3 * n // 4].mean() - ref_mean_dur) / (ref_std_dur + 0.01))
    feats.append(q4 / (q1 + 0.01))

    # Speech rate (2)
    total_time = words[-1]["end_sec"] - words[0]["start_sec"]
    wps = len(words) / (total_time + 0.01)
    feats.append((wps - ref_wps_mean) / (ref_wps_std + 0.01))
    seg_rates = [len(durations[i:i + 10]) / max(
        words[min(i + 9, len(words) - 1)]["end_sec"] - words[i]["start_sec"], 0.5
    ) for i in range(0, len(words) - 10, 5)]
    feats.append(np.std(seg_rates) if seg_rates else 0)

    # Pause counts (5)
    n_words = max(len(words), 1)
    feats.append(len(pauses) / n_words * 100)
    feats.append(sum(1 for s in pauses if s.get("interval_sec", 0) > 0.5) / n_words * 100)
    for th in [1.0, 2.0, 3.0]:
        feats.append(sum(1 for s in pauses if s.get("interval_sec", 0) > th))

    # Disfluency (2)
    feats.append(len(disfluencies) / n_words)
    feats.append(len(disfluencies) / n_words * 100)

    # Silence ratio (1)
    total_pause = pause_durs.sum() if len(pauses) > 0 else 0
    feats.append(total_pause / (total_time + 0.01))

    # Duration by word length (6)
    for bucket in [0, 1, 2, 3, 4, 5]:
        mask = np.array([len(s.get("word", "")) // 2 == bucket for s in words])
        if mask.sum() > 0:
            bm = durations[mask].mean()
            feats.append((bm - ref_mean_dur) / (ref_std_dur + 0.01))
        else:
            feats.append(0)

    # Inter-word intervals (4)
    intervals = []
    for j in range(len(whisper_segments) - 1):
        gap = whisper_segments[j + 1]["start_sec"] - whisper_segments[j]["end_sec"]
        if gap > 0:
            intervals.append(gap)
    intervals = np.array(intervals) if intervals else np.zeros(1)
    feats.append(intervals.mean())
    feats.append(intervals.std() / (intervals.mean() + 0.01))
    feats.append(np.percentile(intervals, 90) if len(intervals) > 0 else 0)
    feats.append(intervals.max() if len(intervals) > 0 else 0)

    # Rhythm regularity (1)
    if len(durations) > 1:
        feats.append(np.abs(np.diff(durations)).mean() / (durations.mean() + 0.01))
    else:
        feats.append(0)

    # Longest fluent segment (3)
    fluent = []
    cur = 0
    for s in whisper_segments:
        if not s.get("is_pause"):
            cur += 1
        else:
            if cur > 0:
                fluent.append(cur)
                cur = 0
    if cur > 0:
        fluent.append(cur)
    if fluent:
        feats.extend([max(fluent), np.mean(fluent), np.std(fluent) / (np.mean(fluent) + 0.01)])
    else:
        feats.extend([0, 0, 0])

    # Pause clustering (1)
    pause_pos = np.array([i for i, s in enumerate(whisper_segments) if s.get("is_pause")])
    feats.append(np.diff(pause_pos).std() if len(pause_pos) > 1 else 0)

    feats = np.array(feats[:60], dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    if len(feats) < 60:
        feats = np.pad(feats, (0, 60 - len(feats)))
    return feats


def predict(wav_path, verbose=True):
    """
    Predict AD probability from a WAV file.

    Args:
        wav_path: Path to .wav file (16kHz mono recommended).
        verbose: Print progress messages.

    Returns:
        dict with keys: probability, prediction, transcript, pause_analysis.
    """
    _load_models()
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load audio
    if verbose:
        print(f"Loading: {wav_path}")
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=16000)
    audio = audio.astype(np.float32)

    # Whisper ASR (local model, no network)
    if verbose:
        print("Running Whisper ASR...")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    whisper_path = str(MODEL_DIR / "whisper-medium")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_path, local_files_only=True)
    processor_w = WhisperProcessor.from_pretrained(whisper_path, local_files_only=True)
    whisper_model = whisper_model.to(_device)

    input_features = processor_w(audio, sampling_rate=16000, return_tensors="pt").input_features.to(_device)
    with torch.no_grad():
        generated_ids = whisper_model.generate(
            input_features, language="english", task="transcribe",
            return_timestamps=True, max_new_tokens=384,
        )

    result_w = processor_w.decode(generated_ids[0], output_offsets=True, skip_special_tokens=False)
    transcript = result_w.get("text", "").strip()

    # Distribute words evenly within each sentence segment
    PAUSE_GAP_SEC = 0.2
    raw_words = []
    if "offsets" in result_w and result_w["offsets"]:
        for seg in result_w["offsets"]:
            text = seg["text"].strip()
            ts = seg.get("timestamp", (0, 0))
            seg_start = float(ts[0]) if ts[0] is not None else 0.0
            seg_end = float(ts[1]) if ts[1] is not None else 0.0
            words_in_seg = text.split()
            seg_dur = seg_end - seg_start
            for j, w in enumerate(words_in_seg):
                t_start = seg_start + j * seg_dur / max(1, len(words_in_seg))
                t_end = seg_start + (j + 1) * seg_dur / max(1, len(words_in_seg))
                raw_words.append({"word": w, "start_sec": t_start, "end_sec": t_end})

    if not raw_words:
        words_list = transcript.split()
        dur = len(audio) / 16000
        for i, w in enumerate(words_list):
            raw_words.append({
                "word": w,
                "start_sec": i * dur / max(1, len(words_list)),
                "end_sec": (i + 1) * dur / max(1, len(words_list)),
            })

    del whisper_model
    torch.cuda.empty_cache()

    # Build segment list with pauses and disfluency markers
    whisper_segs = []
    for i, w in enumerate(raw_words):
        if i > 0:
            gap = w["start_sec"] - raw_words[i - 1]["end_sec"]
            if gap > PAUSE_GAP_SEC:
                whisper_segs.append({
                    "word": "[**]",
                    "start_sec": raw_words[i - 1]["end_sec"],
                    "end_sec": w["start_sec"],
                    "interval_sec": gap,
                    "is_disfluent": False,
                    "is_pause": True,
                })

        is_disfl = (
            i > 0 and i < len(raw_words) - 1
            and w["word"].lower() == raw_words[i - 1]["word"].lower()
        )
        interval = w["end_sec"] - w["start_sec"]

        whisper_segs.append({
            "word": w["word"],
            "start_sec": w["start_sec"],
            "end_sec": w["end_sec"],
            "interval_sec": interval,
            "is_disfluent": is_disfl,
            "is_pause": False,
        })

    n_words = sum(1 for s in whisper_segs if not s["is_pause"])
    n_pauses = sum(1 for s in whisper_segs if s["is_pause"])
    if verbose:
        print(f"  Words: {n_words}, Pauses: {n_pauses}")
        print(f"  Transcript: {transcript[:200]}...")

    # Pause statistics (diagnostic, not used in ensemble)
    if verbose:
        print("Extracting pause features...")
    _ = extract_pause_features(whisper_segs, _libri_ref)

    # RoBERTa features
    if verbose:
        print("Extracting RoBERTa features...")
    clean_text = " ".join(
        s["word"] for s in whisper_segs
        if not s.get("is_pause") and not s.get("is_disfluent")
    )
    if not clean_text.strip():
        clean_text = "."
    rob_cls = _extract_roberta_cls(clean_text)

    # 5-fold IW ensemble prediction (RoBERTa-only, 768d)
    if verbose:
        print("Predicting (5-fold ensemble)...")
    probs = []
    for clf_i, scaler_i in zip(_fold_clfs, _fold_scalers):
        X_s = scaler_i.transform(rob_cls)
        probs.append(float(clf_i.predict_proba(X_s)[0, 1]))
    prob = float(np.mean(probs))
    pred = "AD" if prob > 0.5 else "CN"

    # Pause statistics vs healthy reference
    word_durs = np.array([s["end_sec"] - s["start_sec"] for s in whisper_segs if not s.get("is_pause")])
    total_time = whisper_segs[-1]["end_sec"] - whisper_segs[0]["start_sec"] if whisper_segs else 1

    pause_analysis = {
        "avg_word_duration": f"{word_durs.mean():.2f}s" if len(word_durs) > 0 else "N/A",
        "healthy_ref_word_duration": f"{_libri_ref.get('word_duration_mean', 0.3):.2f}s",
        "num_pauses": n_pauses,
        "words_per_second": f"{len(word_durs) / (total_time + 0.01):.1f}",
        "healthy_ref_wps": f"{_libri_ref.get('words_per_sec_mean', 2.8):.1f}",
    }

    if verbose:
        print(f"\n  {'='*40}")
        print(f"  AD Probability: {prob:.1%}")
        print(f"  Prediction: {pred}")
        print(f"  {'='*40}")

    return {
        "probability": prob,
        "prediction": pred,
        "transcript": transcript,
        "pause_analysis": pause_analysis,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict AD probability from a WAV file")
    parser.add_argument("wav_path", help="Path to WAV file")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    result = predict(args.wav_path, verbose=not args.quiet)
    print(f"\nResult: {result['prediction']} (probability: {result['probability']:.1%})")
    print(f"Transcript: {result['transcript'][:300]}...")
    print(f"Pause stats: {result['pause_analysis']}")
