"""
StethoEdge - AI Heart & Lung Auscultation Screening

Models:
    models/heart_normal_abnormal.keras
    models/lung_normal_abnormal.keras
    models/lung_abnormal_type.keras
    models/lung_abnormal_labels.npy

Run:
    python -m streamlit run demo.py

Research/hackathon prototype - not a clinical diagnostic tool.
"""

from pathlib import Path
import hashlib
import html
import os
import tempfile
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf


# ============================================================
# CONFIG
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"

HEART_MODEL_PATH = MODEL_DIR / "heart_normal_abnormal.keras"
LUNG_MODEL_PATH = MODEL_DIR / "lung_normal_abnormal.keras"
LUNG_TYPE_MODEL_PATH = MODEL_DIR / "lung_abnormal_type.keras"
LUNG_LABELS_PATH = MODEL_DIR / "lung_abnormal_labels.npy"

RECORDING_DURATION = 10
MODEL_DURATION = 5
SAMPLE_RATE = 16000
MODEL_SAMPLES = SAMPLE_RATE * MODEL_DURATION

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

# Lenient minimum quality standards for real auscultation clips
MIN_LIVE_DURATION_S = 5.0
MIN_UPLOAD_DURATION_S = 3.0
MIN_RMS = 0.002
MIN_MODEL_WINDOW_RMS = 0.0015
MIN_PEAK = 0.015
MIN_STD = 0.001
MAX_SILENCE_RATIO = 0.98
MAX_CLIPPING_RATIO = 0.05
MIN_QUALITY_SCORE = 50
BORDERLINE_DIFF_THRESHOLD = 0.15

MODEL_SPECS = [
    ("heart", "Heart Screening", HEART_MODEL_PATH, "keras"),
    ("lung", "Lung Screening", LUNG_MODEL_PATH, "keras"),
    ("lung_type", "Lung Sound Classifier", LUNG_TYPE_MODEL_PATH, "keras"),
    ("lung_labels", "Lung Class Labels", LUNG_LABELS_PATH, "npy"),
]

# ---- Edit these for the About page --------------------------
TEAM_NAME = ""                 # e.g. "Team StethoEdge"
TEAM_MEMBERS: list[str] = []   # e.g. ["Member One", "Member Two"]
PROJECT_URL = ""               # e.g. "https://github.com/<you>/stethoedge"
# --------------------------------------------------------------

ORGAN_HISTORY_CONFIG = {
    "Heart": {
        "conditions": [
            "Coronary artery disease / prior MI",
            "Heart murmur / valve disease",
            "Arrhythmia (e.g., AFib)",
            "Heart failure / cardiomyopathy",
            "Hypertension / high cholesterol",
            "Family history of early heart disease",
        ],
        "symptoms": [
            "Chest pain or pressure",
            "Palpitations",
            "Exertional shortness of breath",
            "Dizziness or leg swelling",
        ],
        "red_flags": [
            "Severe chest pain (arm/jaw radiation)",
            "Fainting / syncope",
            "Rest dyspnea",
        ],
        "exposures": ["Current smoker", "Former smoker"],
    },
    "Lungs": {
        "conditions": [
            "Asthma",
            "COPD / chronic bronchitis",
            "Prior pneumonia or TB",
            "Interstitial lung disease",
            "Prior pulmonary embolism",
        ],
        "symptoms": [
            "Persistent cough",
            "Wheezing",
            "Shortness of breath",
            "Sputum or fever",
        ],
        "red_flags": [
            "Severe respiratory distress",
            "Coughing up blood",
            "Cyanosis (blue lips/nails)",
        ],
        "exposures": ["Current smoker / vaper", "Dust or fume exposure"],
    },
}

ORGAN_LABELS = {"Heart": "❤️  Heart", "Lungs": "🫁  Lungs"}
MODE_LABELS = {"Live Recording": "🎙️  Live", "Upload File": "📁  Upload"}


def html_block(markup: str) -> str:
    """
    Strip indentation and blank lines from an HTML snippet.

    st.markdown() follows Markdown rules: a blank line ends an HTML block and
    any line indented by 4+ spaces afterwards becomes a *code block*.
    Flattening the snippet keeps it rendering as HTML.
    """
    return "\n".join(line.strip() for line in markup.splitlines() if line.strip())


def show_html(markup: str) -> None:
    st.markdown(html_block(markup), unsafe_allow_html=True)


# ============================================================
# PAGE CONFIG & STYLING
# ============================================================

st.set_page_config(
    page_title="StethoEdge | AI Heart & Lung Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"], .stMarkdown, button, input, textarea {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 340px at 8% -10%, rgba(14, 165, 233, 0.10), transparent 60%),
        radial-gradient(800px 320px at 95% -6%, rgba(16, 185, 129, 0.08), transparent 55%),
        #090d16;
}

[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stDeployButton"] {
    display: none !important;
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 1.5rem;
    max-width: 1180px;
}
[data-testid="stVerticalBlock"] { gap: 0.85rem; }

/* ---------- Header ---------- */
.nav-bar {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 0.75rem;
    padding-bottom: 0.9rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.12);
}
.nav-brand { display: flex; align-items: center; gap: 0.8rem; }
.nav-logo {
    width: 42px; height: 42px; border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
}
.nav-title { font-size: 1.45rem; font-weight: 800; letter-spacing: -0.02em; color: #f8fafc; line-height: 1.1; }
.nav-sub { font-size: 0.82rem; color: #64748b; margin-top: 0.15rem; }
.nav-tag {
    font-size: 0.72rem; font-weight: 600; color: #94a3b8;
    background: rgba(148, 163, 184, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.16);
    padding: 0.32rem 0.75rem; border-radius: 999px;
}

/* ---------- Tabs ---------- */
[data-baseweb="tab-list"] { gap: 0.3rem; }
button[data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; padding: 0.55rem 1rem; }

/* ---------- Step labels ---------- */
.step-lbl {
    display: flex; align-items: center; gap: 0.55rem;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: #94a3b8;
}
.step-num {
    width: 22px; height: 22px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.74rem; font-weight: 800; color: #04101a;
    background: linear-gradient(135deg, #38bdf8, #34d399);
}
.field-lbl {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #64748b; margin-bottom: 0.3rem;
}

/* ---------- Buttons ---------- */
button[kind="primary"],
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primaryFormSubmit"],
div[data-testid="stFormSubmitButton"] > button {
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #38bdf8, #34d399) !important;
    border: none !important;
    color: #04101a !important;
}
button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
div[data-testid="stFormSubmitButton"] > button:hover { filter: brightness(1.07); }
button[kind="primary"]:disabled,
button[data-testid="stBaseButton-primary"]:disabled { opacity: 0.35; }

/* ---------- Empty / landing state ---------- */
.hero-empty {
    padding: 1.7rem 1.7rem 1.5rem;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.10), rgba(16, 185, 129, 0.06));
    border: 1px solid rgba(56, 189, 248, 0.22);
}
.hero-kicker {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #38bdf8; margin-bottom: 0.5rem;
}
.hero-h { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em; color: #f8fafc; line-height: 1.15; }
.hero-p { font-size: 0.95rem; color: #94a3b8; margin-top: 0.55rem; max-width: 620px; line-height: 1.6; }

.step-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.7rem; margin-top: 1.2rem; }
.step-card {
    padding: 0.85rem 0.95rem; border-radius: 12px;
    background: rgba(9, 13, 22, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.14);
}
.step-card .n { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #38bdf8; font-weight: 700; }
.step-card .t { font-size: 0.92rem; font-weight: 700; color: #f1f5f9; margin-top: 0.25rem; }
.step-card .d { font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem; line-height: 1.5; }

.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.7rem; }
.stat {
    padding: 0.8rem 0.9rem; border-radius: 12px; text-align: center;
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.13);
}
.stat .v { font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: #f8fafc; }
.stat .k { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; color: #64748b; margin-top: 0.15rem; }

.feat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.7rem; }
.feat-card {
    padding: 0.95rem 1rem; border-radius: 12px;
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.13);
}
.feat-card .i { font-size: 1.25rem; }
.feat-card .t { font-size: 0.9rem; font-weight: 700; color: #f1f5f9; margin-top: 0.3rem; }
.feat-card .d { font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem; line-height: 1.5; }

/* ---------- Tips ---------- */
.tip-card {
    padding: 0.95rem 1.05rem; border-radius: 12px;
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.13);
}
.tip-head { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #38bdf8; margin-bottom: 0.45rem; }
.tip-list { margin: 0; padding-left: 1.05rem; color: #cbd5e1; font-size: 0.84rem; line-height: 1.6; }

/* ---------- Verdict ---------- */
.verdict-box {
    padding: 1.3rem 1.4rem; border-radius: 14px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.14);
}
.verdict-row { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.75rem; }
.verdict-word { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.01em; line-height: 1.05; }
.v-norm { color: #34d399; }
.v-abn { color: #f87171; }
.verdict-meta { font-size: 0.8rem; color: #94a3b8; margin-top: 0.3rem; }
.conf-badge {
    font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 700;
    padding: 0.35rem 0.8rem; border-radius: 999px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1); color: #f8fafc;
}
.prob-bar-wrap { margin-top: 0.95rem; }
.prob-labels {
    display: flex; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: #94a3b8; margin-bottom: 0.32rem;
}
.split-track { display: flex; width: 100%; height: 8px; border-radius: 999px; overflow: hidden; background: rgba(148, 163, 184, 0.14); }
.split-norm { background: #34d399; height: 100%; }
.split-abn { background: #f87171; height: 100%; }
.borderline-pill {
    display: inline-block; margin-top: 0.75rem; font-size: 0.76rem; color: #fbbf24;
    background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.28);
    padding: 0.25rem 0.6rem; border-radius: 7px;
}

/* ---------- Clinical guide ---------- */
.kpi-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.65rem; margin-bottom: 0.85rem; }
.kpi-tile { padding: 0.75rem 0.9rem; border-radius: 11px; background: rgba(15, 23, 42, 0.55); border: 1px solid rgba(148, 163, 184, 0.13); }
.kpi-k { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; color: #64748b; }
.kpi-v { font-size: 0.96rem; font-weight: 700; color: #f1f5f9; margin-top: 0.2rem; }
.cue-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
.cue-card { padding: 0.9rem 1rem; border-radius: 12px; background: rgba(15, 23, 42, 0.55); border: 1px solid rgba(148, 163, 184, 0.13); }
.cue-head { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; color: #38bdf8; margin-bottom: 0.45rem; }
.cue-list { margin: 0; padding-left: 1rem; color: #cbd5e1; font-size: 0.84rem; line-height: 1.55; }

/* ---------- Visual explanation note ---------- */
.why-note {
    padding: 0.75rem 0.95rem; border-radius: 10px;
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.15);
    font-size: 0.83rem; color: #cbd5e1; line-height: 1.5;
}

/* ---------- Subtype bars ---------- */
.sub-row { margin-bottom: 0.5rem; }
.sub-meta { display: flex; justify-content: space-between; font-size: 0.82rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.2rem; }
.sub-track { width: 100%; height: 6px; border-radius: 999px; background: rgba(148, 163, 184, 0.14); overflow: hidden; }
.sub-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #38bdf8, #34d399); }

/* ---------- Alerts ---------- */
.q-fail {
    padding: 0.9rem 1.05rem; border-radius: 11px;
    background: rgba(239, 68, 68, 0.10); border: 1px solid rgba(239, 68, 68, 0.35); font-size: 0.9rem;
}
.rec-pill {
    padding: 0.85rem 1rem; border-radius: 11px;
    background: rgba(56, 189, 248, 0.09); border: 1px solid rgba(56, 189, 248, 0.28);
    margin-top: 0.75rem; font-size: 0.88rem; color: #e2e8f0;
}
.rec-warn {
    padding: 0.85rem 1rem; border-radius: 11px; margin-top: 0.85rem;
    background: rgba(245, 158, 11, 0.09); border: 1px solid rgba(245, 158, 11, 0.3);
    font-size: 0.88rem; color: #fde68a;
}

/* ---------- How to use / About ---------- */
.page-h { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; color: #f8fafc; }
.page-p { font-size: 0.95rem; color: #94a3b8; line-height: 1.65; max-width: 760px; }
.how-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; }
.how-card {
    padding: 1.05rem 1.1rem; border-radius: 14px;
    background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.14);
}
.how-card .n {
    width: 28px; height: 28px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 0.82rem; color: #04101a;
    background: linear-gradient(135deg, #38bdf8, #34d399);
}
.how-card .t { font-size: 0.98rem; font-weight: 700; color: #f1f5f9; margin-top: 0.6rem; }
.how-card .d { font-size: 0.83rem; color: #94a3b8; margin-top: 0.3rem; line-height: 1.55; }

.tip-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; }

.pipe { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.6rem; }
.pipe-step {
    padding: 0.85rem 0.8rem; border-radius: 12px; text-align: center;
    background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.14);
}
.pipe-step .i { font-size: 1.3rem; }
.pipe-step .t { font-size: 0.86rem; font-weight: 700; color: #f1f5f9; margin-top: 0.25rem; }
.pipe-step .d { font-size: 0.74rem; color: #94a3b8; margin-top: 0.2rem; line-height: 1.45; }

.model-row {
    display: flex; justify-content: space-between; align-items: center; gap: 0.75rem;
    padding: 0.7rem 0.9rem; border-radius: 10px; margin-bottom: 0.45rem;
    background: rgba(15, 23, 42, 0.55); border: 1px solid rgba(148, 163, 184, 0.13);
}
.model-name { font-size: 0.9rem; font-weight: 600; color: #f1f5f9; }
.model-file { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #64748b; margin-top: 0.1rem; }
.pill-ok, .pill-err {
    font-size: 0.72rem; font-weight: 700; padding: 0.15rem 0.55rem; border-radius: 999px; white-space: nowrap;
}
.pill-ok { color: #34d399; background: rgba(52, 211, 153, 0.12); border: 1px solid rgba(52, 211, 153, 0.35); }
.pill-err { color: #f87171; background: rgba(248, 113, 113, 0.12); border: 1px solid rgba(248, 113, 113, 0.35); }

.chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.chip {
    font-size: 0.78rem; font-weight: 600; color: #cbd5e1;
    padding: 0.3rem 0.75rem; border-radius: 999px;
    background: rgba(148, 163, 184, 0.08); border: 1px solid rgba(148, 163, 184, 0.16);
}

.site-foot {
    margin-top: 0.5rem; padding-top: 1rem;
    border-top: 1px solid rgba(148, 163, 184, 0.12);
    text-align: center; font-size: 0.78rem; color: #64748b; line-height: 1.6;
}

@media (max-width: 900px) {
    .step-grid, .feat-grid, .tip-grid, .kpi-row, .cue-grid { grid-template-columns: 1fr; }
    .how-grid { grid-template-columns: 1fr 1fr; }
    .stat-grid { grid-template-columns: 1fr 1fr; }
    .pipe { grid-template-columns: 1fr; }
    .hero-h { font-size: 1.5rem; }
}
</style>
"""

show_html(CSS)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading models...")
def load_models() -> tuple[dict[str, Any], dict[str, str]]:
    """
    Load every available model. A model that fails to load (for example due
    to a Keras/TensorFlow version mismatch) is reported on the About page
    instead of crashing the whole app.
    """
    loaded: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for key, _, path, kind in MODEL_SPECS:
        if not path.exists():
            continue
        try:
            if kind == "keras":
                loaded[key] = tf.keras.models.load_model(str(path), compile=False)
            elif kind == "npy":
                loaded[key] = np.load(path, allow_pickle=True)
        except Exception as exc:  # noqa: BLE001
            errors[key] = f"{type(exc).__name__}: {exc}"

    return loaded, errors


models, model_errors = load_models()


# ============================================================
# AUDIO, QUALITY & CLINICAL FEATURES
# ============================================================

def load_audio_bytes(audio_bytes: bytes, suffix: str = ".wav") -> tuple[np.ndarray, float]:
    """Decode audio bytes into 16 kHz mono float32 and duration."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return audio.astype(np.float32), len(audio) / SAMPLE_RATE


def build_model_input(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare the 5-second window and 64-band log-Mel tensor.

    Must match the preprocessing used in train_heart_model.py /
    train_lung_model.py (sample rate, n_mels, n_fft, hop_length, ref).
    """
    if len(audio) < MODEL_SAMPLES:
        model_audio = np.pad(audio, (0, MODEL_SAMPLES - len(audio)))
    else:
        model_audio = audio[:MODEL_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=model_audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    X = mel_db.astype(np.float32)[np.newaxis, ..., np.newaxis]
    return model_audio, mel_db, X


def assess_audio_quality(audio: np.ndarray, duration: float, input_mode: str) -> dict[str, Any]:
    """Enforce lenient minimum acoustic quality before inference."""
    if len(audio) == 0:
        return {"acceptable": False, "score": 0, "duration": 0.0, "reasons": ["No audio signal captured."]}

    min_duration = MIN_LIVE_DURATION_S if input_mode == "Live Recording" else MIN_UPLOAD_DURATION_S
    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    model_rms = float(np.sqrt(np.mean(np.square(audio[:MODEL_SAMPLES])) + 1e-12))
    peak = float(np.max(np.abs(audio)))
    std_dev = float(np.std(audio))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.99))
    silence_ratio = float(np.mean(np.abs(audio) < max(0.002, peak * 0.03)))

    reasons: list[str] = []
    score = 100

    if duration < min_duration:
        reasons.append(f"Clip too short ({duration:.1f}s, minimum {min_duration:.0f}s).")
        score -= 40
    if rms < MIN_RMS:
        reasons.append("Signal level is too faint.")
        score -= 35
    if model_rms < MIN_MODEL_WINDOW_RMS:
        reasons.append("The first 5 seconds contain almost no signal.")
        score -= 30
    if peak < MIN_PEAK or std_dev < MIN_STD:
        reasons.append("Dead or flat microphone input.")
        score -= 25
    if silence_ratio > MAX_SILENCE_RATIO:
        reasons.append("Audio is nearly 100% silent.")
        score -= 30
    if clipping_ratio > MAX_CLIPPING_RATIO:
        reasons.append(f"Heavy distortion ({clipping_ratio * 100:.1f}% clipped).")
        score -= 30

    score = max(0, min(100, score))
    return {
        "acceptable": len(reasons) == 0 and score >= MIN_QUALITY_SCORE,
        "score": score,
        "duration": duration,
        "reasons": reasons,
    }


def compute_clinical_profile(model_audio: np.ndarray) -> dict[str, Any]:
    """Dominant pitch and Bell vs. Diaphragm stethoscope guidance."""
    if len(model_audio) == 0:
        return {"dom_hz": 0.0, "side": "Bell / Diaphragm", "band_note": "n/a", "low_pct": 0.0}

    spectrum = np.abs(np.fft.rfft(model_audio * np.hanning(len(model_audio)))) ** 2
    freqs = np.fft.rfftfreq(len(model_audio), d=1.0 / SAMPLE_RATE)
    mask = (freqs >= 20) & (freqs <= 2000)
    total = float(np.sum(spectrum[mask])) + 1e-12

    low_pct = (float(np.sum(spectrum[(freqs >= 20) & (freqs < 200)])) / total) * 100.0
    sub_freqs = freqs[mask]
    sub_spec = spectrum[mask]
    dom_hz = float(sub_freqs[int(np.argmax(sub_spec))]) if len(sub_spec) > 0 else 0.0

    is_low = dom_hz < 200 or low_pct >= 65.0
    side = "Bell (20–200 Hz)" if is_low else "Diaphragm (200–2k Hz)"
    band_note = f"{low_pct:.0f}% low-freq" if is_low else f"{100 - low_pct:.0f}% mid/high-freq"
    return {"dom_hz": dom_hz, "side": side, "band_note": band_note, "low_pct": low_pct}


# ============================================================
# INFERENCE & VISUAL EXPLANATION POINTERS
# ============================================================

def predict_binary(model: Any, X: np.ndarray) -> tuple[str, float, float, float]:
    """
    Binary normal/abnormal classification.

    Assumes class 0 = normal, class 1 = abnormal for 2-unit outputs and a
    single sigmoid output = P(abnormal). Verify against your training labels.
    """
    prediction = np.asarray(model.predict(X, verbose=0)[0]).reshape(-1)
    if prediction.size < 2:
        abnormal_prob = float(prediction[0])
        normal_prob = 1.0 - abnormal_prob
    else:
        normal_prob = float(prediction[0])
        abnormal_prob = float(prediction[1])

    if abnormal_prob >= normal_prob:
        return "ABNORMAL", abnormal_prob, normal_prob, abnormal_prob
    return "NORMAL", normal_prob, normal_prob, abnormal_prob


def predict_lung_type(X: np.ndarray) -> tuple[str, float, np.ndarray]:
    """Classify abnormal lung sound subtype."""
    prediction = np.asarray(models["lung_type"].predict(X, verbose=0)[0]).reshape(-1)
    labels = models["lung_labels"]
    idx = int(np.argmax(prediction))
    return str(labels[idx]), float(prediction[idx]), prediction


def compute_model_pointers(
    model: Any,
    X: np.ndarray,
    model_audio: np.ndarray,
    label: str,
) -> list[dict[str, float]]:
    """
    Locate the top 2 time-frequency moments in the 5s window that most
    strongly drove the model's predicted class (via gradient saliency, with
    an acoustic-energy fallback if gradients are unavailable).
    """
    time_curve: np.ndarray | None = None
    try:
        x_tf = tf.convert_to_tensor(X, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            preds = model(x_tf, training=False)
            if preds.shape[-1] is not None and preds.shape[-1] >= 2:
                target_idx = 1 if label == "ABNORMAL" else 0
                score = preds[0, target_idx]
            else:
                score = preds[0, 0] if label == "ABNORMAL" else (1.0 - preds[0, 0])
        grads = tape.gradient(score, x_tf)
        if grads is not None:
            sal = np.abs(grads[0, :, :, 0].numpy())
            if np.max(sal) > 0:
                time_curve = sal.mean(axis=0)
    except Exception:  # noqa: BLE001
        time_curve = None

    if time_curve is None:
        # Fallback: frame-wise energy on the Mel spectrogram
        time_curve = np.mean(np.maximum(X[0, :, :, 0] + 80.0, 0.0), axis=0)

    # Smooth over ~0.25s and trim edges so annotations sit inside the plot
    kernel = np.ones(7, dtype=np.float32) / 7.0
    smooth = np.convolve(time_curve, kernel, mode="same")
    n_frames = len(smooth)
    edge = max(4, int(0.30 / MODEL_DURATION * n_frames))
    min_sep = max(10, int(1.0 / MODEL_DURATION * n_frames))

    work = smooth.copy()
    work[:edge] = 0.0
    work[-edge:] = 0.0

    pointers: list[dict[str, float]] = []
    peak_ref = float(np.max(work)) if len(work) > 0 else 0.0
    if peak_ref <= 0:
        return pointers

    for _ in range(2):
        idx = int(np.argmax(work))
        val = float(work[idx])
        if val < 0.35 * peak_ref:
            break

        t_sec = (idx / max(1, n_frames - 1)) * MODEL_DURATION
        t0 = max(0.0, t_sec - 0.25)
        t1 = min(float(MODEL_DURATION), t_sec + 0.25)

        s0 = int(t0 * SAMPLE_RATE)
        s1 = int(t1 * SAMPLE_RATE)
        seg = model_audio[s0:s1]
        if len(seg) > 32:
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
            freqs = np.fft.rfftfreq(len(seg), d=1.0 / SAMPLE_RATE)
            mask = (freqs >= 20) & (freqs <= 2000)
            sub_f = freqs[mask]
            sub_s = spec[mask]
            hz = float(sub_f[int(np.argmax(sub_s))]) if len(sub_s) > 0 else 120.0
        else:
            hz = 120.0

        pointers.append({"t": t_sec, "t0": t0, "t1": t1, "hz": hz})

        lo = max(0, idx - min_sep)
        hi = min(n_frames, idx + min_sep + 1)
        work[lo:hi] = 0.0

    return sorted(pointers, key=lambda p: p["t"])


def format_label(raw: str) -> str:
    return str(raw).replace("_", " ").title()


def clamp_prob(val: float) -> float:
    return min(max(float(val), 0.0), 1.0)


def reset_results() -> None:
    st.session_state.analysis = None
    st.session_state.rejection = None


def run_screening(
    audio_bytes: bytes,
    suffix: str,
    organ: str,
    input_mode: str,
    clip_name: str,
) -> None:
    """Quality check + inference. Stores the outcome in session state."""
    reset_results()

    audio, duration = load_audio_bytes(audio_bytes, suffix=suffix)
    quality = assess_audio_quality(audio, duration, input_mode)

    if not quality["acceptable"]:
        st.session_state.rejection = quality
        return

    model_key = "heart" if organ == "Heart" else "lung"
    if model_key not in models:
        raise RuntimeError(
            f"The {organ.lower()} model is missing or failed to load. "
            "See the About page for model status."
        )

    model_audio, _, X = build_model_input(audio)
    label, conf, norm_p, abn_p = predict_binary(models[model_key], X)
    pointers = compute_model_pointers(models[model_key], X, model_audio, label)

    l_type = l_conf = l_probs = None
    if organ == "Lungs" and label == "ABNORMAL" and "lung_type" in models and "lung_labels" in models:
        l_type, l_conf, l_probs = predict_lung_type(X)

    st.session_state.analysis = {
        "screening_type": organ,
        "input_mode": input_mode,
        "clip_name": clip_name,
        "quality": quality,
        "audio": audio,
        "model_audio": model_audio,
        "label": label,
        "confidence": conf,
        "normal_p": norm_p,
        "abnormal_p": abn_p,
        "lung_type": l_type,
        "lung_type_conf": l_conf,
        "lung_type_probs": l_probs,
        "pointers": pointers,
    }


# ============================================================
# PLOTS (WITH VISUAL POINTERS)
# ============================================================

def render_spectrogram_figure(
    model_audio: np.ndarray,
    label: str = "NORMAL",
    pointers: list[dict[str, float]] | None = None,
) -> plt.Figure:
    """Clean 20-2000 Hz clinical frequency map with model-focus pointers."""
    stft_mag = np.abs(librosa.stft(model_audio, n_fft=2048, hop_length=256))
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=2048)
    mask = (freqs >= 20) & (freqs <= 2000)
    spec_db = librosa.amplitude_to_db(stft_mag[mask, :], ref=np.max)

    fig, ax = plt.subplots(figsize=(6.8, 3.0), dpi=140)
    fig.patch.set_facecolor("#090d16")
    ax.set_facecolor("#090d16")

    img = ax.imshow(spec_db, origin="lower", aspect="auto", cmap="magma", extent=[0, MODEL_DURATION, 20, 2000])
    ax.axhline(200, color="#38bdf8", linestyle="--", linewidth=0.8, alpha=0.65)
    ax.axhline(600, color="#34d399", linestyle="--", linewidth=0.8, alpha=0.65)

    accent = "#34d399" if label == "NORMAL" else "#f87171"
    for idx, ptr in enumerate(pointers or [], start=1):
        t, hz = ptr["t"], ptr["hz"]
        ax.axvspan(ptr["t0"], ptr["t1"], color=accent, alpha=0.14)
        ax.scatter([t], [hz], color=accent, edgecolors="#090d16", linewidth=1.2, s=55, zorder=5)
        y_off = -210 if hz > 1450 else 155
        va = "top" if hz > 1450 else "bottom"
        ha = "right" if t > 3.6 else ("left" if t < 1.0 else "center")
        ax.annotate(
            f"#{idx} {t:.1f}s • {hz:.0f} Hz",
            xy=(t, hz),
            xytext=(t, np.clip(hz + y_off, 90, 1910)),
            color="#f8fafc",
            fontsize=7.5,
            fontweight="bold",
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#0f172a", edgecolor=accent, alpha=0.9),
            arrowprops=dict(arrowstyle="-", color=accent, lw=0.9),
            zorder=6,
        )

    ax.set_title("Frequency Map (20–2000 Hz)", color="#e2e8f0", fontsize=10, pad=8)
    ax.set_xlabel("Time (s)", color="#94a3b8", fontsize=8.5)
    ax.set_ylabel("Hz", color="#94a3b8", fontsize=8.5)
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")

    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.tick_params(colors="#64748b", labelsize=7.5)
    cbar.outline.set_edgecolor("#1e293b")
    fig.tight_layout()
    return fig


def render_waveform_figure(
    audio: np.ndarray,
    label: str = "NORMAL",
    pointers: list[dict[str, float]] | None = None,
) -> plt.Figure:
    """Waveform + amplitude envelope with linked model-focus pointers."""
    fig, ax = plt.subplots(figsize=(6.8, 3.0), dpi=140)
    fig.patch.set_facecolor("#090d16")
    ax.set_facecolor("#090d16")

    dur = len(audio) / SAMPLE_RATE
    t = np.linspace(0, dur, num=len(audio))
    ax.plot(t, audio, color="#38bdf8", linewidth=0.65, alpha=0.85)
    ax.fill_between(t, audio, 0, color="#38bdf8", alpha=0.14)

    env = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
    ax.plot(np.linspace(0, dur, num=len(env)), env * 2.1, color="#fbbf24", linewidth=1.1)

    max_amp = float(np.max(np.abs(audio)) or 0.1)
    y_bound = max(max_amp * 1.22, float(np.max(env * 2.1)) * 1.12, 0.05)
    ax.set_ylim(-max_amp * 1.22, y_bound)

    if dur > MODEL_DURATION + 0.2:
        ax.axvspan(0, MODEL_DURATION, color="#38bdf8", alpha=0.05)
        ax.axvline(MODEL_DURATION, color="#38bdf8", linestyle=":", linewidth=0.9, alpha=0.65)
        ax.text(
            MODEL_DURATION + 0.12,
            y_bound * 0.92,
            "5s window",
            color="#7dd3fc",
            fontsize=7.2,
            va="top",
        )

    accent = "#34d399" if label == "NORMAL" else "#f87171"
    for idx, ptr in enumerate(pointers or [], start=1):
        ax.axvspan(ptr["t0"], ptr["t1"], color=accent, alpha=0.20)
        s0 = int(ptr["t0"] * SAMPLE_RATE)
        s1 = int(ptr["t1"] * SAMPLE_RATE)
        seg = audio[s0:s1]
        if len(seg) > 0:
            local_idx = int(np.argmax(np.abs(seg)))
            pt_t = ptr["t0"] + local_idx / SAMPLE_RATE
            pt_y = float(seg[local_idx])
            ax.scatter([pt_t], [pt_y], color=accent, edgecolors="#090d16", linewidth=1.1, s=45, zorder=5)
            ha = "left" if pt_t < dur * 0.78 else "right"
            y_txt = pt_y + (0.14 * max_amp if pt_y >= 0 else -0.14 * max_amp)
            ax.annotate(
                f"#{idx}",
                xy=(pt_t, pt_y),
                xytext=(pt_t, y_txt),
                color="#f8fafc",
                fontsize=7.5,
                fontweight="bold",
                ha=ha,
                va="bottom" if pt_y >= 0 else "top",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#0f172a", edgecolor=accent, alpha=0.9),
                zorder=6,
            )

    ax.set_title("Cadence & Envelope", color="#e2e8f0", fontsize=10, pad=8)
    ax.set_xlabel("Time (s)", color="#94a3b8", fontsize=8.5)
    ax.set_xlim(0, max(MODEL_DURATION, dur))
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")

    fig.tight_layout()
    return fig


# ============================================================
# HEADER & FOOTER
# ============================================================

def render_header() -> None:
    show_html(
        """
        <div class="nav-bar">
        <div class="nav-brand">
        <div class="nav-logo">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M2 12h3l2.5-6 3.5 12 3-8 2 2h6"/>
        </svg>
        </div>
        <div>
        <div class="nav-title">StethoEdge</div>
        <div class="nav-sub">AI-assisted cardiac &amp; pulmonary auscultation</div>
        </div>
        </div>
        <div class="nav-tag">Prototype • Not for clinical diagnosis</div>
        </div>
        """
    )


def render_footer() -> None:
    show_html(
        """
        <div class="site-foot">
        StethoEdge is an experimental research and educational prototype. Results are not a medical diagnosis
        and must never replace examination by a qualified healthcare professional.<br>
        Built with Streamlit • TensorFlow/Keras • librosa
        </div>
        """
    )


# ============================================================
# SCREENING TAB
# ============================================================

def step_label(num: int, text: str) -> None:
    show_html(f'<div class="step-lbl"><span class="step-num">{num}</span>{html.escape(text)}</div>')


def pick(key: str, label: str, options: list[str], default: str, labels: dict[str, str]) -> str:
    """Segmented control that never returns None (falls back to last choice)."""
    val = st.segmented_control(
        label,
        options,
        default=default,
        key=key,
        format_func=lambda o: labels[o],
        label_visibility="collapsed",
    )
    if val is None:
        val = st.session_state.get(f"{key}_last", default)
    st.session_state[f"{key}_last"] = val
    return val


def render_empty_state() -> None:
    ready = sum(1 for key in ("heart", "lung", "lung_type") if key in models)
    show_html(
        f"""
        <div class="hero-empty">
        <div class="hero-kicker">AI-assisted auscultation screening</div>
        <div class="hero-h">Listen smarter. Screen in seconds.</div>
        <div class="hero-p">Record or upload a heart or lung sound and get an instant normal / abnormal screen,
        a frequency map and bedside guidance on what to do next.</div>
        <div class="step-grid">
        <div class="step-card"><div class="n">STEP 1</div><div class="t">Choose organ</div><div class="d">Heart or lungs, each with its own model and guidance.</div></div>
        <div class="step-card"><div class="n">STEP 2</div><div class="t">Add audio</div><div class="d">Record live or upload a WAV, MP3, FLAC or OGG clip.</div></div>
        <div class="step-card"><div class="n">STEP 3</div><div class="t">Run screening</div><div class="d">Quality check first, then the AI model on the first {MODEL_DURATION} s.</div></div>
        </div>
        </div>
        <div class="stat-grid">
        <div class="stat"><div class="v">{MODEL_DURATION} s</div><div class="k">Analysis window</div></div>
        <div class="stat"><div class="v">{SAMPLE_RATE // 1000} kHz</div><div class="k">Sample rate</div></div>
        <div class="stat"><div class="v">{N_MELS}</div><div class="k">Mel bands</div></div>
        <div class="stat"><div class="v">{ready}/3</div><div class="k">Models ready</div></div>
        </div>
        <div class="feat-grid">
        <div class="feat-card"><div class="i">🎯</div><div class="t">Verdict &amp; confidence</div><div class="d">Normal or abnormal with a probability split and a borderline warning.</div></div>
        <div class="feat-card"><div class="i">📊</div><div class="t">Frequency map</div><div class="d">Spectrogram and waveform so you can see what the model heard.</div></div>
        <div class="feat-card"><div class="i">🩺</div><div class="t">Bedside guide</div><div class="d">Bell vs. diaphragm advice, exam cues and a short history check.</div></div>
        </div>
        """
    )


def render_rejection(quality: dict[str, Any]) -> None:
    reasons = " ".join(html.escape(r) for r in quality["reasons"])
    show_html(
        f"""
        <div class="q-fail"><strong>Recording rejected (quality {quality["score"]}/100)</strong><br>{reasons}</div>
        """
    )
    show_html(
        """
        <div class="tip-card">
        <div class="tip-head">Try again</div>
        <ul class="tip-list">
        <li>Hold the microphone or stethoscope flat and still against the skin.</li>
        <li>Move to a quiet spot and avoid talking during the recording.</li>
        <li>Record for at least 5 seconds (10 seconds is ideal).</li>
        </ul>
        </div>
        """
    )


def render_verdict_card(analysis: dict[str, Any]) -> None:
    """Single-box verdict + split probability bar."""
    organ = analysis["screening_type"]
    label = analysis["label"]
    norm_pct = clamp_prob(analysis["normal_p"]) * 100
    abn_pct = clamp_prob(analysis["abnormal_p"]) * 100
    conf_pct = clamp_prob(analysis["confidence"]) * 100
    diff = abs(norm_pct - abn_pct)

    word_cls = "v-norm" if label == "NORMAL" else "v-abn"
    clip_label = html.escape(analysis["clip_name"])
    q_score = analysis["quality"]["score"]
    dur = analysis["quality"]["duration"]

    borderline = (
        f'<div class="borderline-pill">Borderline margin ({diff:.1f} pts) - verify clinically</div>'
        if diff < BORDERLINE_DIFF_THRESHOLD * 100
        else ""
    )

    show_html(
        f"""
        <div class="verdict-box">
        <div class="verdict-row">
        <div>
        <div class="verdict-word {word_cls}">{html.escape(label)}</div>
        <div class="verdict-meta">{html.escape(organ)} • {clip_label} • {dur:.1f}s • Quality {q_score}/100</div>
        </div>
        <div class="conf-badge">{conf_pct:.1f}% Confidence</div>
        </div>
        <div class="prob-bar-wrap">
        <div class="prob-labels"><span>Normal {norm_pct:.1f}%</span><span>Abnormal {abn_pct:.1f}%</span></div>
        <div class="split-track">
        <div class="split-norm" style="width:{norm_pct:.1f}%;"></div>
        <div class="split-abn" style="width:{abn_pct:.1f}%;"></div>
        </div>
        </div>
        {borderline}
        </div>
        """
    )

    if organ == "Lungs" and label == "ABNORMAL" and analysis["lung_type"] is not None:
        with st.container(border=True):
            show_html('<div class="field-lbl">Detected Lung Sound Pattern</div>')
            ranked = sorted(
                zip(models["lung_labels"], analysis["lung_type_probs"]),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            bars = []
            for name, prob in ranked:
                pct = clamp_prob(prob) * 100
                bars.append(
                    f'<div class="sub-row">'
                    f'<div class="sub-meta"><span>{html.escape(format_label(name))}</span><span>{pct:.1f}%</span></div>'
                    f'<div class="sub-track"><div class="sub-fill" style="width:{pct:.1f}%;"></div></div>'
                    f'</div>'
                )
            show_html("".join(bars))


def render_trainee_guide(analysis: dict[str, Any]) -> None:
    """Concise, high-signal bedside cues plus history check / next step."""
    organ = analysis["screening_type"]
    label = analysis["label"]
    lung_type = format_label(analysis["lung_type"]).lower() if analysis.get("lung_type") else ""
    prof = compute_clinical_profile(analysis["model_audio"])

    if organ == "Heart":
        exam_focus = "APETM • Pulse Sync"
        bedside_cues = [
            "<li>Palpate carotid pulse to separate systole vs. diastole</li>",
            "<li>Apex (5th L ICS) with Bell; Base (2nd R/L ICS) with Diaphragm</li>",
            "<li>Check radiation to carotids or axilla</li>",
        ]
        watch_out = (
            [
                "<li>Normal clip doesn't exclude subtle diastolic murmurs</li>",
                "<li>Try left lateral decubitus (MS/S3) &amp; lean forward (AR)</li>",
                "<li>Always correlate with vitals &amp; history below</li>",
            ]
            if label == "NORMAL"
            else [
                "<li>Rule out fever, anemia, or tachycardia flow murmurs</li>",
                "<li>Eliminate chest-hair rub or clothing friction</li>",
                "<li>Confirm timing clinically; order echo if persistent</li>",
            ]
        )
    else:
        exam_focus = "Symmetric Ladder"
        if label == "NORMAL":
            bedside_cues = [
                "<li>Compare Left vs. Right at each level (anterior &amp; posterior)</li>",
                "<li>Ask for deep, open-mouth breaths</li>",
                "<li>Percuss bases if patient is dyspneic or febrile</li>",
            ]
            watch_out = [
                "<li>Single 5s spot can miss focal early pneumonia</li>",
                "<li>Check SpO₂, respiratory rate, and tactile fremitus</li>",
                "<li>Review lung history &amp; exposures below</li>",
            ]
        elif "wheez" in lung_type and "crackl" not in lung_type:
            bedside_cues = [
                "<li>Continuous musical sound (400–1600 Hz) from narrowed airways</li>",
                "<li>Note expiratory vs. inspiratory phase &amp; polyphonic vs. focal</li>",
                "<li>Check SpO₂ &amp; work of breathing immediately</li>",
            ]
            watch_out = [
                "<li>Consider asthma, COPD flare, or focal obstruction</li>",
                "<li>Severe obstruction can produce a 'silent chest'</li>",
                "<li>Re-assess air entry after bronchodilator</li>",
            ]
        else:
            bedside_cues = [
                "<li>Fine late crackles = edema/fibrosis; coarse = secretions</li>",
                "<li>Ask patient to cough &amp; re-listen to see if sounds clear</li>",
                "<li>Compare symmetric zones top-to-bottom</li>",
            ]
            watch_out = [
                "<li>Rule out chest-hair friction or tubing noise</li>",
                "<li>Correlate with fever, SpO₂, and percussion dullness</li>",
                "<li>Consider chest imaging if focal signs persist</li>",
            ]

    side = html.escape(prof["side"])
    band = html.escape(prof["band_note"])
    focus = html.escape(exam_focus)
    cues_html = "".join(bedside_cues)
    watch_html = "".join(watch_out)

    show_html(
        f"""
        <div class="kpi-row">
        <div class="kpi-tile"><div class="kpi-k">Use Stethoscope</div><div class="kpi-v">{side}</div></div>
        <div class="kpi-tile"><div class="kpi-k">Dominant Pitch</div><div class="kpi-v">{prof["dom_hz"]:.0f} Hz ({band})</div></div>
        <div class="kpi-tile"><div class="kpi-k">Bedside Focus</div><div class="kpi-v">{focus}</div></div>
        </div>
        <div class="cue-grid">
        <div class="cue-card"><div class="cue-head">Bedside Action</div><ul class="cue-list">{cues_html}</ul></div>
        <div class="cue-card"><div class="cue-head">Clinical Watch-Out</div><ul class="cue-list">{watch_html}</ul></div>
        </div>
        """
    )

    if label != "NORMAL":
        show_html(
            """
            <div class="rec-warn"><strong>Abnormal pattern flagged.</strong>
            Have a qualified clinician review this patient. This is a model classification,
            not a diagnosis.</div>
            """
        )
        return

    cfg = ORGAN_HISTORY_CONFIG[organ]
    st.markdown("---")
    show_html(f'<div class="field-lbl">{html.escape(organ)} medical history check (normal screen)</div>')

    with st.form(f"history_{organ.lower()}"):
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            age = st.number_input("Age", min_value=0, max_value=120, value=25, step=1)
            conditions = st.multiselect(f"Prior {organ.lower()} history", cfg["conditions"])
            exposures = st.multiselect("Exposure", cfg["exposures"])
        with c2:
            symptoms = st.multiselect("Current symptoms", cfg["symptoms"])
            red_flags = st.multiselect("Red flags", cfg["red_flags"])

        submitted = st.form_submit_button("Check next step", type="primary", width="stretch")

    if submitted:
        if red_flags:
            msg = f"URGENT EVALUATION - {organ} red-flag symptoms override a normal AI clip."
        elif symptoms or conditions or exposures:
            msg = f"CLINICAL REVIEW - Discuss {organ.lower()} history, symptoms, or exposures with a senior clinician."
        else:
            msg = f"ROUTINE CARE - No {organ.lower()} risk factors reported (age {age})."
        show_html(f'<div class="rec-pill">{html.escape(msg)}</div>')


def render_pointer_summary(analysis: dict[str, Any]) -> None:
    """Short 1-line explanation of what the highlighted pointers show."""
    pointers = analysis.get("pointers") or []
    organ = analysis["screening_type"]
    label = analysis["label"]
    prof = compute_clinical_profile(analysis["model_audio"])

    if not pointers:
        return

    spots = " and ".join(f"<strong>{p['t']:.1f}s ({p['hz']:.0f} Hz)</strong>" for p in pointers)
    if organ == "Heart":
        if label == "NORMAL":
            reason = f"regular low-frequency bursts ({prof['low_pct']:.0f}% in 20–200 Hz) consistent with steady S1/S2 beats."
        else:
            reason = f"irregular or extra acoustic energy ({prof['band_note']}, peak ~{prof['dom_hz']:.0f} Hz) pushing the model toward ABNORMAL."
    else:
        if label == "NORMAL":
            reason = f"even breath-cycle energy ({prof['band_note']}) without prominent adventitious spikes."
        else:
            subtype = format_label(analysis["lung_type"]) if analysis.get("lung_type") else "abnormal"
            reason = f"concentrated {subtype.lower()} acoustic energy ({prof['band_note']}, peak ~{prof['dom_hz']:.0f} Hz)."

    show_html(
        f"""
        <div class="why-note">
        <strong>Why {html.escape(label)}:</strong> Highlighted regions (#1–#{len(pointers)}) at {spots}
        had the strongest influence on this prediction — showing {reason}
        </div>
        """
    )


def render_results(analysis: dict[str, Any]) -> None:
    render_verdict_card(analysis)

    tab_guide, tab_spec = st.tabs(["Clinical guide", "Audio & spectrum"])

    with tab_guide:
        render_trainee_guide(analysis)

    with tab_spec:
        pointers = analysis.get("pointers") or []
        label = analysis["label"]
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            fig_s = render_spectrogram_figure(analysis["model_audio"], label=label, pointers=pointers)
            st.pyplot(fig_s)
            plt.close(fig_s)
        with c2:
            fig_w = render_waveform_figure(analysis["audio"], label=label, pointers=pointers)
            st.pyplot(fig_w)
            plt.close(fig_w)
        render_pointer_summary(analysis)
        st.caption(
            "Highlighted bands mark the model's top sensitivity windows. Dashed lines mark 200 Hz "
            "(Bell / Diaphragm) and 600 Hz; only the first 5 seconds are analysed."
        )


def render_screening_tab() -> None:
    if not models:
        st.error(
            "No models were found in the `models/` folder, so screening is unavailable. "
            "See the About page for details."
        )

    left_col, right_col = st.columns([0.85, 1.6], gap="large")

    audio_bytes: bytes | None = None
    audio_suffix = ".wav"
    clip_name = "Live Mic"
    run_clicked = False
    organ = "Heart"
    input_mode = "Live Recording"

    # ---------------- LEFT: controls ----------------
    with left_col:
        with st.container(border=True):
            step_label(1, "Select organ")
            organ = pick("organ_pick", "Organ", list(ORGAN_LABELS), "Heart", ORGAN_LABELS)
            if st.session_state.last_organ != organ:
                st.session_state.last_organ = organ
                reset_results()

            step_label(2, "Add audio")
            input_mode = pick("mode_pick", "Audio input", list(MODE_LABELS), "Live Recording", MODE_LABELS)
            if st.session_state.last_mode != input_mode:
                st.session_state.last_mode = input_mode
                st.session_state.audio_id = None
                reset_results()

            if input_mode == "Live Recording":
                try:
                    rec = st.audio_input("Record 5–10 seconds", sample_rate=SAMPLE_RATE)
                except TypeError:
                    rec = st.audio_input("Record 5–10 seconds")
                if rec is not None:
                    audio_bytes = rec.getvalue()
                    clip_name = "Live Mic"
            else:
                up = st.file_uploader(
                    "Upload WAV / MP3 / FLAC / OGG",
                    type=["wav", "mp3", "flac", "ogg"],
                    label_visibility="collapsed",
                )
                if up is not None:
                    audio_bytes = up.getvalue()
                    audio_suffix = Path(up.name).suffix.lower() or ".wav"
                    clip_name = up.name
                    st.audio(up)

            step_label(3, "Run")
            if audio_bytes is not None:
                cid = hashlib.sha256(audio_bytes).hexdigest()
                if st.session_state.audio_id != cid:
                    st.session_state.audio_id = cid
                    reset_results()
                run_clicked = st.button("Run screening", type="primary", width="stretch")
            else:
                st.session_state.audio_id = None
                reset_results()
                st.button("Run screening", type="primary", width="stretch", disabled=True)
                st.caption("Add a recording or file to enable screening.")

        show_html(
            """
            <div class="tip-card">
            <div class="tip-head">Quick recording tips</div>
            <ul class="tip-list">
            <li>Quiet room, no talking, fans off.</li>
            <li>Mic or stethoscope flat and firm on bare skin.</li>
            <li>Hold still for 8–10 seconds.</li>
            <li>Lungs: breathe slowly and deeply through the mouth.</li>
            </ul>
            </div>
            """
        )

    # ---------------- RIGHT: results ----------------
    with right_col:
        if run_clicked and audio_bytes is not None:
            try:
                with st.spinner("Screening audio..."):
                    run_screening(audio_bytes, audio_suffix, organ, input_mode, clip_name)
            except Exception as exc:  # noqa: BLE001
                reset_results()
                st.error("Unable to process this audio.")
                st.exception(exc)

        rejection = st.session_state.rejection
        analysis = st.session_state.analysis

        if rejection is not None:
            render_rejection(rejection)
        elif analysis is not None:
            render_results(analysis)
        elif audio_bytes is not None:
            st.info("Clip loaded. Press **Run screening** to analyse it.")
            render_empty_state()
        else:
            render_empty_state()


# ============================================================
# HOW TO USE TAB
# ============================================================

def render_how_to_use() -> None:
    show_html(
        """
        <div class="page-h">How to use StethoEdge</div>
        <div class="page-p">Four steps from recording to result. No account or setup is needed,
        and everything runs in your browser session.</div>
        <div class="how-grid">
        <div class="how-card"><span class="n">1</span><div class="t">Choose the organ</div><div class="d">Pick <strong>Heart</strong> or <strong>Lungs</strong>. This selects the matching AI model and clinical guidance.</div></div>
        <div class="how-card"><span class="n">2</span><div class="t">Add a recording</div><div class="d">Record live with your microphone or upload a WAV, MP3, FLAC or OGG file of a stethoscope recording.</div></div>
        <div class="how-card"><span class="n">3</span><div class="t">Run the screening</div><div class="d">A quality check runs first. If the clip passes, the first 5 seconds go through the AI model.</div></div>
        <div class="how-card"><span class="n">4</span><div class="t">Read the result</div><div class="d">Review the verdict, confidence, spectrum and bedside guide. Normal results add a short history check.</div></div>
        </div>
        """
    )

    show_html('<div class="page-h" style="font-size:1.15rem;">Getting a good recording</div>')
    show_html(
        """
        <div class="tip-grid">
        <div class="tip-card"><div class="tip-head">Environment</div><ul class="tip-list">
        <li>Choose a quiet room and switch off fans or music.</li>
        <li>Ask everyone to stay silent while recording.</li>
        <li>Avoid rustling clothes or tapping the device.</li>
        </ul></div>
        <div class="tip-card"><div class="tip-head">Placement</div><ul class="tip-list">
        <li><strong>Heart:</strong> apex (left 5th intercostal space) or left sternal border.</li>
        <li><strong>Lungs:</strong> back of the chest, either side of the spine.</li>
        <li>Press flat and firm on bare skin; avoid chest-hair rub.</li>
        </ul></div>
        <div class="tip-card"><div class="tip-head">Recording</div><ul class="tip-list">
        <li>Record for 8–10 seconds and hold still.</li>
        <li>The model uses the first 5 seconds, so start on the sound.</li>
        <li>Lungs: slow, deep breaths through an open mouth.</li>
        </ul></div>
        </div>
        """
    )

    show_html('<div class="page-h" style="font-size:1.15rem;">Understanding the result</div>')
    show_html(
        """
        <div class="tip-grid">
        <div class="tip-card"><div class="tip-head">Verdict &amp; confidence</div><ul class="tip-list">
        <li><strong>Normal / Abnormal</strong> is the class the model rates most likely.</li>
        <li>Confidence is the model's probability, not a clinical certainty.</li>
        <li>A <em>borderline</em> warning appears when the two probabilities are close.</li>
        </ul></div>
        <div class="tip-card"><div class="tip-head">Lung sound pattern</div><ul class="tip-list">
        <li>For abnormal lung clips a second model ranks the likely sound type.</li>
        <li>Treat it as a hint for the examiner, not a diagnosis.</li>
        </ul></div>
        <div class="tip-card"><div class="tip-head">Clinical guide</div><ul class="tip-list">
        <li>Suggests Bell or Diaphragm from the dominant pitch.</li>
        <li>Lists bedside actions and watch-outs for the result.</li>
        <li>Normal results include a quick history and red-flag check.</li>
        </ul></div>
        </div>
        """
    )

    show_html('<div class="page-h" style="font-size:1.15rem;">Frequently asked questions</div>')

    with st.expander("Why was my recording rejected?"):
        st.markdown(
            f"""
Before the AI runs, a technical quality check looks for:

- **Length:** at least {MIN_LIVE_DURATION_S:.0f} s for live recordings and {MIN_UPLOAD_DURATION_S:.0f} s for uploads.
- **Signal level:** not too faint, and not a dead or flat microphone input.
- **Silence:** the clip must not be almost entirely silent, including the first {MODEL_DURATION} s the model reads.
- **Clipping:** no more than {MAX_CLIPPING_RATIO * 100:.0f}% of samples may be distorted.

Re-record closer to the chest in a quieter place and try again.
            """
        )

    with st.expander("Which audio files can I upload?"):
        st.markdown(
            "WAV, MP3, FLAC and OGG. Clips are converted to 16 kHz mono automatically. "
            "Clean stethoscope recordings work best; phone-microphone clips of room sound will not."
        )

    with st.expander("The microphone does not start"):
        st.markdown(
            "- Allow microphone access when your browser asks, or in the site settings next to the address bar.\n"
            "- Live recording needs a secure (HTTPS) page or `localhost`.\n"
            "- Close other apps that may be using the microphone.\n"
            "- As a fallback, record with any app and use **Upload**."
        )

    with st.expander("Is my audio stored?"):
        st.markdown(
            "The app writes your clip to a temporary file only to decode it and deletes it straight away. "
            "Nothing is saved by the app itself."
        )

    with st.expander("Can this replace a doctor?"):
        st.markdown(
            "No. StethoEdge is a research prototype that has not been clinically validated. "
            "It is meant for demonstration and education, and any concerning symptom needs a real medical evaluation."
        )


# ============================================================
# ABOUT TAB
# ============================================================

def render_about() -> None:
    show_html(
        """
        <div class="page-h">About StethoEdge</div>
        <div class="page-p">StethoEdge is an AI-assisted screening prototype for heart and lung sounds.
        It turns a short stethoscope recording into a normal / abnormal screen, shows what the model
        "heard" as a spectrogram, and gives practical bedside guidance. It was built as a hackathon project
        to explore how low-cost audio and deep learning could support early screening where specialists
        are scarce.</div>
        """
    )

    show_html('<div class="page-h" style="font-size:1.15rem;">How it works</div>')
    show_html(
        f"""
        <div class="pipe">
        <div class="pipe-step"><div class="i">🎙️</div><div class="t">Capture</div><div class="d">Live mic or uploaded file, converted to {SAMPLE_RATE // 1000} kHz mono.</div></div>
        <div class="pipe-step"><div class="i">✅</div><div class="t">Quality gate</div><div class="d">Checks length, level, silence and clipping before any prediction.</div></div>
        <div class="pipe-step"><div class="i">📈</div><div class="t">Log-Mel features</div><div class="d">First {MODEL_DURATION} s turned into a {N_MELS}-band spectrogram image.</div></div>
        <div class="pipe-step"><div class="i">🧠</div><div class="t">CNN models</div><div class="d">Keras networks classify normal vs. abnormal, then lung sound type.</div></div>
        <div class="pipe-step"><div class="i">🩺</div><div class="t">Guidance</div><div class="d">Verdict, confidence, frequency map and bedside tips.</div></div>
        </div>
        """
    )

    show_html('<div class="page-h" style="font-size:1.15rem;">Models</div>')

    rows = []
    for key, name, path, _ in MODEL_SPECS:
        if key in models:
            pill = '<span class="pill-ok">Ready</span>'
        elif key in model_errors:
            pill = '<span class="pill-err">Failed to load</span>'
        else:
            pill = '<span class="pill-err">Missing</span>'
        rows.append(
            f'<div class="model-row"><div><div class="model-name">{html.escape(name)}</div>'
            f'<div class="model-file">models/{html.escape(path.name)}</div></div>{pill}</div>'
        )
    show_html("".join(rows))

    for key, err in model_errors.items():
        st.caption(f"{key}: {err[:300]}")

    if "lung_labels" in models:
        names = [format_label(n) for n in models["lung_labels"]]
        chips = "".join(f'<span class="chip">{html.escape(n)}</span>' for n in names)
        show_html(f'<div class="field-lbl">Lung sound classes</div><div class="chip-row">{chips}</div>')

    show_html('<div class="page-h" style="font-size:1.15rem;">Responsible use</div>')
    show_html(
        """
        <div class="tip-grid">
        <div class="tip-card"><div class="tip-head">Not a diagnosis</div><ul class="tip-list">
        <li>Outputs are statistical estimates from a prototype model.</li>
        <li>A normal result does not rule out disease.</li>
        </ul></div>
        <div class="tip-card"><div class="tip-head">Limitations</div><ul class="tip-list">
        <li>Trained on limited public data and not clinically validated.</li>
        <li>Only the first 5 seconds of each clip are analysed.</li>
        <li>Noise, poor placement and different devices can change results.</li>
        </ul></div>
        <div class="tip-card"><div class="tip-head">When to seek care</div><ul class="tip-list">
        <li>Chest pain, fainting or severe breathlessness need urgent medical attention.</li>
        <li>Symptoms always outweigh a normal AI result.</li>
        </ul></div>
        </div>
        """
    )

    tech = ["Streamlit", "TensorFlow / Keras", "librosa", "NumPy", "Matplotlib", "Python"]
    chips = "".join(f'<span class="chip">{html.escape(t)}</span>' for t in tech)
    show_html(f'<div class="page-h" style="font-size:1.15rem;">Built with</div><div class="chip-row">{chips}</div>')

    if TEAM_MEMBERS or TEAM_NAME or PROJECT_URL:
        show_html('<div class="page-h" style="font-size:1.15rem;">Team</div>')
        if TEAM_NAME:
            st.markdown(f"**{TEAM_NAME}**")
        if TEAM_MEMBERS:
            chips = "".join(f'<span class="chip">{html.escape(m)}</span>' for m in TEAM_MEMBERS)
            show_html(f'<div class="chip-row">{chips}</div>')
        if PROJECT_URL:
            st.markdown(f"[Project repository]({PROJECT_URL})")


# ============================================================
# APP
# ============================================================

for _key in ("analysis", "rejection", "audio_id", "last_organ", "last_mode"):
    st.session_state.setdefault(_key, None)

render_header()

tab_screen, tab_how, tab_about = st.tabs(["🩺  Screening", "📘  How to use", "ℹ️  About"])

with tab_screen:
    render_screening_tab()

with tab_how:
    render_how_to_use()

with tab_about:
    render_about()

render_footer()
