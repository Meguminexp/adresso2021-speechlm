from faster_whisper import WhisperModel
import collections
import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad

from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QPoint

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QTextEdit,
    QProgressBar,
    QVBoxLayout,
    QHBoxLayout
)

from PySide6.QtGui import (
    QPixmap,
    QFont
)

from PySide6.QtCore import Qt

import sys
import time

FS = 16000
SECONDS = 10
WAV_FILE = "demo_recording.wav"

print("Loading Whisper model...")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("Whisper is ready.")

def record_audio_vad(
    filename="demo_recording.wav",
    fs=16000,
    frame_ms=30,
    max_seconds=10,
    min_record_seconds=4,
    min_speech_seconds=1.2,
    silence_seconds=1.0, # Open day adjustment: 途中で考える人がおおければ 1.3へ
    vad_level=1,         # Open day adjustment: 1 or 2
    rms_threshold=500    # Open day adjustment: うるさければ 600 - 800
):
    vad = webrtcvad.Vad(vad_level)

    frame_size = int(fs * frame_ms / 1000)
    max_frames = int(max_seconds * 1000 / frame_ms)
    silence_frames = int(silence_seconds * 1000 / frame_ms)

    print("Listening with VAD... Please speak.")

    recorded = []
    silence_count = 0
    speech_started = False
    speech_frames = 0
    frames_after_start = 0

    with sd.InputStream(
        samplerate=fs,
        channels=1,
        dtype="int16",
        blocksize=frame_size
    ) as stream:

        for _ in range(max_frames):
            frame, overflowed = stream.read(frame_size)

            pcm = frame.tobytes()

            # VAD + 音量の両方で判定
            rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
            is_loud_enough = rms > rms_threshold
            is_speech = vad.is_speech(pcm, fs) and is_loud_enough

            if is_speech:
                if not speech_started:
                    print(f"Speech detected (RMS={rms:.0f})")

                speech_started = True
                speech_frames += 1
                silence_count = 0
                recorded.append(frame.copy())

            else:
                if speech_started:
                    silence_count += 1
                    recorded.append(frame.copy())

            if speech_started:
                frames_after_start += 1

                record_time = frames_after_start * frame_ms / 1000
                speech_time = speech_frames * frame_ms / 1000

                if record_time >= min_record_seconds and silence_count >= silence_frames:
                    print(f"Record time = {record_time:.1f} sec")
                    print(f"Speech time = {speech_time:.1f} sec")
                    break

    if not speech_started:
        print("No speech detected.")
        return "NO_SPEECH"

    speech_time = speech_frames * frame_ms / 1000
    record_time = frames_after_start * frame_ms / 1000

    print(f"Record time = {record_time:.1f} sec")
    print(f"Speech time = {speech_time:.1f} sec")

    if speech_time < min_speech_seconds:
        print("Speech too short.")
        return "TOO_SHORT"  

    audio = np.concatenate(recorded, axis=0)
    audio_float = audio.astype(np.float32) / 32768.0
    sf.write(filename, audio_float, fs)

    print("VAD recording finished.")
    return filename

def transcribe_audio(filename):
    segments, info = model.transcribe(filename, language="en")
    print(f"Language = {info.language}")
    print(f"Language probability = {info.language_probability:.2f}")
    text = ""
    logprobs = []

    for segment in segments:
        text += segment.text
        logprobs.append(segment.avg_logprob)

    if len(logprobs) > 0:
        avg_logprob = sum(logprobs) / len(logprobs)
    else:
        avg_logprob = -99

    return text.strip(), avg_logprob

def quality_check(text, avg_logprob):

    import re

    text = text.strip()
    words = text.split()
    print("Transcript:")
    print(text)
    BAD_PHRASES = {
        "hello",
        "hi",
        "thank you",
        "thank you very much",
        "good morning",
        "good afternoon",
        "good evening"
    }
    alpha = sum(c.isalpha() for c in text)
    unique_words = len(set(w.lower() for w in words))

    print("----- Transcript QC -----")
    print(f"Text length = {len(text)}")
    print(f"Words = {len(words)}")
    print(f"Unique words = {unique_words}")
    print(f"Alpha = {alpha}")
    print(f"Average log probability = {avg_logprob:.2f}")

    ## special treatment for cokie_thef
    COOKIE_WORDS = {
        "boy",
        "girl",
        "mother",
        "woman",
        "man",
        "cookies",
        "cookie",
        "jar",
        "kitchen",
        "sink",
        "water",
        "plate",
        "window",
        "chair",
        "stool",
        "dish",
        "dishes",
        "wash",
        "washing",
        "fall",
        "falling",
        "dog",
        "table",
        "picture",
        "vase",
        "flower"
        "pen",
        "crayon",
        "pan",
        "pot",
        "bird",
        "towel",
        "fruit",
        "fruits"
    }

    picture_count = 0

    for w in words:
        w = w.lower().strip(".,!?")
        if w in COOKIE_WORDS:
            picture_count += 1

    print(f"Picture words = {picture_count}")

    

    #from collections import Counter
    #counter = Counter(w.lower() for w in words)
    #most_common = counter.most_common(1)[0][1]
    #print(f"Most common count = {most_common}")


    if len(words) < 6:
        print("QC failed: too few words")
        return False

    if len(text) < 20:
        print("QC failed: text too short")
        return False

    if alpha < 10:
        print("QC failed: too few alphabetic characters")
        return False

    if unique_words < 4:
        print("QC failed: too few unique words")
        return False
    
    #if most_common >= len(words) * 0.7:
    #    print("QC failed: too much repetition")
    #    return False
    
    clean_text = (
        text.lower()
            .replace(".", "")
            .replace(",", "")
            .replace("!", "")
            .replace("?", "")
            .strip()
    )

    if len(words) < 6:

        for phrase in BAD_PHRASES:
            if clean_text == phrase:
                print(f"QC failed: only '{phrase}'")
                return False
    
    if avg_logprob < -1.3:
        print("QC failed: low Whisper confidence")
        return False

    #compact_text = text.replace(" ", "")

    #if re.fullmatch(r"(.)\1{10,}", compact_text):
    #    print("QC failed: repeated single character")
    #    return False
    
    if picture_count < 2:
        print("QC failed: not describing the picture")
        return False

    print("QC passed")
    print("-------------------------")

    return True

def predict(wav_file):
    """Run AD detection on recorded audio using the 5-fold IW ensemble."""

    import predict as ad_predict

    result = ad_predict.predict(wav_file, verbose=False)

    # Print results to console for debugging
    print(f"\n  AD Probability: {result['probability']:.1%}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Transcript: {result['transcript'][:200]}...")

    return result["probability"]

class LikelihoodIndicator(QWidget):
    def __init__(self):
        super().__init__()
        self.value = 0.0
        self.setMinimumHeight(55)

    def setValue(self, p):
        self.value = max(0.0, min(1.0, float(p)))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        left = 80
        right = w - 80
        y = h // 2

        pen = QPen(QColor("#B0B0B0"))
        pen.setWidth(4)
        painter.setPen(pen)

        painter.drawLine(left, y, right, y)
        painter.drawLine(left, y - 14, left, y + 14)
        painter.drawLine(right, y - 14, right, y + 14)

        x = left + int((right - left) * self.value)

        painter.setBrush(QBrush(QColor("#2E86DE")))
        painter.setPen(QPen(QColor("#2E86DE")))
        painter.drawEllipse(QPoint(x, y), 11, 11)

def update_indicator(p):
    indicator.setValue(p)

def start_recording():

    button.setEnabled(False)
    transcript_box.clear()
    update_indicator(0.0)
    result_label.setText("Waiting for speech input...")
    result_label.setStyleSheet("""
    QLabel{
        color:white;
        font-weight:bold;
    }
    """)        
    status.setText("🔴 Recording...")
    button.repaint()
    status.repaint()
    QApplication.processEvents()

    try:
        wav = record_audio_vad(
            max_seconds=10,
            min_record_seconds=4,
            min_speech_seconds=0.8,
            silence_seconds=1.0,
            vad_level=1,
            rms_threshold=500
        )

        if wav == "NO_SPEECH":

            status.setText("⚠ No speech detected.")
            result_label.setText("Please speak.")

            button.setEnabled(True)
            return

        if wav == "TOO_SHORT":

            status.setText("⚠ Speech too short.")
            result_label.setText(
                "Please speak for a little longer."
            )

            button.setEnabled(True)
            return

        status.setText("🟠 Transcribing...")
        button.repaint()
        status.repaint()
        QApplication.processEvents()

        try:
            text, avg_logprob = transcribe_audio(wav)
        except Exception as e:
            print("Transcription error:", e)
            text = "(Transcription failed.)"
            avg_logprob = -99

        transcript_box.setPlainText(text)

        if not quality_check(text, avg_logprob):
            result_label.setStyleSheet("""
            QLabel{
                color:#FFCC00;
                font-weight:bold;
            }
            """)

            result_label.setText(
                "Please describe the picture in more detail."
            )

            status.setText("⚠ Please try again.")
            button.setEnabled(True)
            return

        # ---------- QC Passed ----------
        
        result_label.setStyleSheet("""  
        QLabel{
            color:#00BFFF;
            font-weight:bold;
        }
        """)

        result_label.setText("Running AI assessment...")
        result_label.repaint()
        
        button.repaint()
        
        QApplication.processEvents()

        # time.sleep(10)

        try:
            p = predict(wav)
            update_indicator(p)
            result_label.setStyleSheet("""
            QLabel{
                color:#00FF66;
                font-weight:bold;
            }       
            """)
            result_label.setText("AI Assessment Completed")
            status.setText("✅ Finished")
        except Exception as e:
            print("Prediction error:", e)
            update_indicator(0.0)
            result_label.setText("AI assessment unavailable")
            status.setText("⚠ Finished with warning")

    except Exception as e:
        print("Unexpected error:", e)
        status.setText("❌ Error")
        result_label.setText("Please restart or try again.")

    finally:
        button.setEnabled(True)

app = QApplication(sys.argv)

window = QWidget()

layout = QVBoxLayout()
layout.setContentsMargins(30, 10, 30, 20)
layout.setSpacing(8)

logo = QLabel()

pixmap = QPixmap("images/2.png")

pixmap = pixmap.scaled(
    600,
    400,
    Qt.KeepAspectRatio,
    Qt.SmoothTransformation
)

logo.setPixmap(pixmap)
logo.setAlignment(Qt.AlignCenter)

logo.setContentsMargins(0, 0, 0, 0)

layout.addWidget(logo)

# title
# title = QLabel("Dementia Speech AI Demonstrator")
# title = QLabel("Cognitive Decline Detection Demonstration using Speech Language Model")
title = QLabel(
    'Cognitive Decline Detection Demonstration using '
    '<span style="color:#FFD700;"><i><b>Speech Language Model</b></i></span>'
)

title.setTextFormat(Qt.RichText)
title.setAlignment(Qt.AlignCenter)
title.setFont(QFont("Arial", 44, QFont.Bold))

subtitle = QLabel("Research Prototype")
subtitle.setAlignment(Qt.AlignCenter)
subtitle.setFont(QFont("Arial", 24))

layout.addWidget(title)
layout.addWidget(subtitle)

# Speech description picture
picture = QLabel()

pix = QPixmap("images/lima_speech_picture.png")

pix = pix.scaled(
    700,
    500,
    Qt.KeepAspectRatio,
    Qt.SmoothTransformation
)

picture.setPixmap(pix)
picture.setAlignment(Qt.AlignCenter)

layout.addWidget(picture)

# instruction_label
instruction_label = QLabel(
    "Please describe everything you see in the picture for about 5–10 seconds."
)

instruction_label.setAlignment(Qt.AlignCenter)

instruction_label.setFont(QFont("Arial", 24))

instruction_label.setStyleSheet("""
QLabel{
    color:white;
}
""")

layout.addWidget(instruction_label)

# button
button = QPushButton("🎤 Start Recording")

button.setFont(QFont("Arial", 24))

button.setMinimumHeight(60)

layout.addWidget(button)

button.setStyleSheet("""
QPushButton{
    background-color:#1976D2;
    color:white;
    border-radius:10px;
    font-size:26px;
    padding:15px;
}
QPushButton:hover{
    background-color:#1565C0;
}
""")

button.clicked.connect(start_recording)

status = QLabel("🟢 Ready")
status.setAlignment(Qt.AlignCenter)
status.setFont(QFont("Arial",24))

layout.addWidget(status)

### assessment_label
assessment_label = QLabel("AI Cognitive Decline Assessment")

assessment_label.setAlignment(Qt.AlignCenter)

assessment_label.setFont(QFont("Arial",32,QFont.Bold))

layout.addWidget(assessment_label)

#### direction_label
direction_layout = QHBoxLayout()

low_label = QLabel("Lower likelihood")
low_label.setFont(QFont("Arial", 24, QFont.Bold))
low_label.setStyleSheet("color:white;")

high_label = QLabel("Higher likelihood")
high_label.setFont(QFont("Arial", 24, QFont.Bold))
high_label.setStyleSheet("color:white;")

direction_layout.addStretch(1)
direction_layout.addWidget(low_label)
direction_layout.addStretch(3)
direction_layout.addWidget(high_label)
direction_layout.addStretch(1)

layout.addLayout(direction_layout)


### indicator_label
indicator = LikelihoodIndicator()
layout.addWidget(indicator)

### result_label
result_label = QLabel("Waiting for speech input...")

result_label.setAlignment(Qt.AlignCenter)

result_label.setFont(QFont("Arial",24,QFont.Bold))

layout.addWidget(result_label)


transcript_box = QTextEdit()

transcript_box.setReadOnly(True)

transcript_box.setPlaceholderText(
    "Please describe everything you see in the picture..."
)

transcript_box.setMinimumHeight(100)

transcript_box.setStyleSheet("""
QTextEdit{
    background:white;
    color:black;
    font-size:20px;
    border-radius:8px;
    padding:10px;
}
""")

layout.addWidget(transcript_box)

window.setLayout(layout)

window.resize(1500, 900)

window.setWindowTitle("LIMA Dementia Speech AI Demonstrator")

window.show()

sys.exit(app.exec())

