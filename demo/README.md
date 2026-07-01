# AD Detection Demo

Predicts Alzheimer's Disease probability from a spoken Cookie Theft picture description (WAV file).

## Setup

```bash
cd demo/
pip install -r requirements.txt
python3 download_models.py        # first time only (~3.3 GB)
```

## Usage

```bash
python3 predict.py example/sample.wav
python3 predict.py path/to/your/audio.wav
```

**GUI Demo:**
```bash
python3 demo.py
```
Click "Start Recording" and describe the displayed picture for 10-15 seconds in English.

Input: WAV, any sample rate (mono recommended, stereo is averaged).  
Output: AD probability (0-100%), prediction (AD/CN), transcript, pause statistics.

## What's Inside

```
demo/
├── predict.py               CLI prediction (5-fold IW ensemble)
├── demo.py                  GUI demo with microphone recording
├── download_models.py       One-time model downloader
├── retrain_classifier.py    Retrain the classifier (optional)
├── requirements.txt
├── README.md
├── example/
│   ├── sample.wav           Sample from the Cookie Theft task
│   └── test_cases/          Trimmed PAR-only test samples
└── models/
    ├── roberta-base/        RoBERTa (478 MB, downloaded)
    ├── whisper-medium/      Whisper ASR (2.8 GB, downloaded)
    └── classifier/          5-fold IW ensemble (123 KB)
```

## Model

### Method: Seg-Filter + Importance-Weighted 5-Fold Logistic Regression Ensemble

**Seg-Filter (Participant-Only Transcripts)**: The ADReSSo test audio contains both examiner (INV) and participant (PAR) speech. We transcribe the full audio to preserve natural speech rhythm, then use the provided speaker-segmentation files to filter out examiner sentences — keeping only participant speech for RoBERTa encoding. This removes a spurious feature: examiner instruction echoing, which AD patients do more often but which contaminates the purely linguistic signal.

**Importance Weighting (IW)**: Training (166 subjects) and test (71 subjects) come from slightly different distributions. IW estimates a weight $w(\mathbf{x}) = P_{\text{test}}(\mathbf{x}) / P_{\text{train}}(\mathbf{x})$ for each training sample via a domain classifier that distinguishes training from test samples. A weighted Logistic Regression then up-weights training samples that resemble the test distribution and down-weights those that don't. This corrects for covariate shift without touching test labels.

**5-Fold Ensemble**: Five separate LR models are trained on different CV splits, each with its own per-fold StandardScaler and IW weights. At inference, all five models vote and their predicted probabilities are averaged. This reduces variance compared to a single model.

### Performance

Tested on ADReSSo 2021 (71 held-out subjects: 35 AD / 36 CN):

| Metric | Value |
|--------|-------|
| Accuracy | **85.92%** |
| F1 | **85.71%** |
| AUC | **90.24%** |
| False Positives | 5 / 36 (13.9%) |
| False Negatives | 5 / 35 (14.3%) |

### Features

- **RoBERTa-base CLS token (768d)**: Encodes linguistic patterns — word-finding difficulty, repetition, simplified syntax, idea density.
- **Pause statistics** (60d, diagnostic only): Word duration percentiles, speech rate, pause frequency — compared against LibriSpeech healthy-speaker norms. Displayed for interpretation but not used in the ensemble prediction.

## Output Example

```
Loading RoBERTa...
Loading 5-fold IW ensemble...
  5 folds loaded on cpu
Loading: example/sample.wav
Running Whisper ASR...
  Words: 65, Pauses: 0
  Transcript: Everything. Everything you see happening...
Extracting RoBERTa features...
Predicting (5-fold ensemble)...

  ========================================
  AD Probability: 77.3%
  Prediction: AD
  ========================================

Result: AD (probability: 77.3%)
```

## Limitations

- Model is trained on Cookie Theft picture descriptions only. Other tasks (animal naming, story recall) will produce unreliable results.
- Best performance with 15+ seconds of participant speech. Very short recordings (<5 words) are ambiguous.
- PAUSE features are shown for diagnostic purposes but not used in the ensemble prediction.
