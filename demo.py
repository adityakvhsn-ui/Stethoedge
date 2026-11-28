"""
StethoEdge - AI Heart & Lung Auscultation Screening

Models:
    models/heart_normal_abnormal.keras
    models/lung_normal_abnormal.keras
    models/lung_abnormal_type.keras
    models/lung_abnormal_labels.npy

Run:
    python -m streamlit run demo.py

Research/hackathon prototype — not a clinical diagnostic tool.
"""

from pathlib import Path
import hashlib
import html
import os
import tempfile
from textwrap import dedent
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


# ============================================================
# PAGE CONFIG & MINIMALIST STYLING
# ============================================================

st.set_page_config(
    page_title="StethoEdge",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    dedent(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(850px 320px at 10% -8%, rgba(14, 165, 233, 0.09), transparent 60%),
                radial-gradient(750px 300px at 90% -5%, rgba(16, 185, 129, 0.07), transparent 55%),
                #090d16;
        }

        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stDeployButton"], .stDeployButton, .stAppDeployButton, #MainMenu, footer {
            display: none !important;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        /* Minimal Header */
        .nav-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 1rem;
            margin-bottom: 1.2rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        }

        .nav-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .nav-logo {
            width: 38px;
            height: 38px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.28);
        }

        .nav-title {
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #f8fafc;
            margin: 0;
            line-height: 1.1;
        }

        .nav-sub {
            font-size: 0.8rem;
            color: #64748b;
            margin: 0.1rem 0 0 0;
        }

        .nav-tag {
            font-size: 0.72rem;
            font-weight: 600;
            color: #94a3b8;
            background: rgba(148, 163, 184, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.16);
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
        }

        /* Segmented Pills */
        div[role="radiogroup"] {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.45rem;
        }

        div[role="radiogroup"] > label {
            justify-content: center;
            padding: 0.52rem 0.65rem !important;
            border-radius: 9px !important;
            border: 1px solid rgba(148, 163, 184, 0.16) !important;
            background: rgba(255, 255, 255, 0.02) !important;
            margin-right: 0 !important;
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            transition: all 0.15s ease;
        }

        div[role="radiogroup"] > label:hover {
            border-color: rgba(56, 189, 248, 0.45) !important;
        }

        div[role="radiogroup"] > label:has(input:checked) {
            border-color: #38bdf8 !important;
            background: rgba(56, 189, 248, 0.14) !important;
            color: #f8fafc !important;
        }

        div[role="radiogroup"] > label > div:first-of-type {
            display: none !important;
        }

        .field-lbl {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.35rem;
        }

        /* Primary CTA */
        div.stButton > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 10px !important;
            padding: 0.58rem 1rem !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, #38bdf8, #34d399) !important;
            border: none !important;
            color: #04101a !important;
        }

        div.stButton > button[kind="primary"]:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            filter: brightness(1.07);
        }

        /* Verdict Hero */
        .verdict-box {
            padding: 1.3rem 1.4rem;
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.14);
            margin-bottom: 0.85rem;
        }

        .verdict-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.75rem;
        }

        .verdict-word {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            line-height: 1.05;
        }

        .v-norm { color: #34d399; }
        .v-abn { color: #f87171; }

        .verdict-meta {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 0.3rem;
        }

        .conf-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 700;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #f8fafc;
        }

        .prob-bar-wrap {
            margin-top: 0.95rem;
        }

        .prob-labels {
            display: flex;
            justify-content: space-between;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            color: #94a3b8;
            margin-bottom: 0.32rem;
        }

        .split-track {
            display: flex;
            width: 100%;
            height: 8px;
            border-radius: 999px;
            overflow: hidden;
            background: rgba(148, 163, 184, 0.14);
        }

        .split-norm { background: #34d399; height: 100%; }
        .split-abn { background: #f87171; height: 100%; }

        .borderline-pill {
            display: inline-block;
            margin-top: 0.75rem;
            font-size: 0.76rem;
            color: #fbbf24;
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.28);
            padding: 0.25rem 0.6rem;
            border-radius: 7px;
        }

        /* Compact Clinical KPI Strip */
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.65rem;
            margin-bottom: 0.85rem;
        }

        .kpi-tile {
            padding: 0.75rem 0.9rem;
            border-radius: 11px;
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.13);
        }

        .kpi-k {
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: #64748b;
        }

        .kpi-v {
            font-size: 0.96rem;
            font-weight: 700;
            color: #f1f5f9;
            margin-top: 0.2rem;
        }

        /* Compact 2-Card Bedside Guide */
        .cue-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
        }

        .cue-card {
            padding: 0.9rem 1rem;
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.13);
        }

        .cue-head {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: #38bdf8;
            margin-bottom: 0.45rem;
        }

        .cue-list {
            margin: 0;
            padding-left: 1rem;
            color: #cbd5e1;
            font-size: 0.84rem;
            line-height: 1.55;
        }

        /* Subtype Bars */
        .sub-row { margin-bottom: 0.5rem; }
        .sub-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.82rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 0.2rem;
        }
        .sub-track {
            width: 100%;
            height: 6px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.14);
            overflow: hidden;
        }
        .sub-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #38bdf8, #34d399);
        }

        /* Empty & Alert Cards */
        .calm-empty {
            padding: 3.4rem 1.5rem;
            border-radius: 14px;
            border: 1px dashed rgba(148, 163, 184, 0.22);
            background: rgba(15, 23, 42, 0.28);
            text-align: center;
            color: #94a3b8;
        }

        .q-fail {
            padding: 0.85rem 1rem;
            border-radius: 11px;
            background: rgba(239, 68, 68, 0.10);
            border: 1px solid rgba(239, 68, 68, 0.35);
            margin-bottom: 0.75rem;
            font-size: 0.9rem;
        }

        .rec-pill {
            padding: 0.85rem 1rem;
            border-radius: 11px;
            background: rgba(56, 189, 248, 0.09);
            border: 1px solid rgba(56, 189, 248, 0.28);
            margin-top: 0.75rem;
            font-size: 0.88rem;
            color: #e2e8f0;
        }

        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.45rem 0.65rem;
            border-radius: 8px;
            background: rgba(148, 163, 184, 0.07);
            margin-bottom: 0.35rem;
            font-size: 0.82rem;
        }
        .dot-ok { color: #34d399; font-weight: 700; font-size: 0.72rem; }
        .dot-err { color: #f87171; font-weight: 700; font-size: 0.72rem; }

        @media (max-width: 768px) {
            .kpi-row, .cue-grid { grid-template-columns: 1fr; }
        }
        </style>
        """
    ).strip(),
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models() -> dict[str, Any]:
    """Load available Keras models and label arrays."""
    loaded: dict[str, Any] = {}
    for key, _, path, kind in MODEL_SPECS:
        if not path.exists():
            continue
        if kind == "keras":
            loaded[key] = tf.keras.models.load_model(path)
        elif kind == "npy":
            loaded[key] = np.load(path)
    return loaded


models = load_models()


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
    """Prepare 5-second audio window and 64-band log-Mel spectrogram tensor."""
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


def assess_audio_quality(
    audio: np.ndarray,
    duration: float,
    input_mode: str,
) -> dict[str, Any]:
    """Enforce lenient minimum acoustic quality before inference."""
    if len(audio) == 0:
        return {
            "acceptable": False,
            "score": 0,
            "duration": 0.0,
            "reasons": ["No audio signal captured."],
        }

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
        reasons.append(f"Clip too short ({duration:.1f}s < {min_duration:.0f}s minimum).")
        score -= 40
    if rms < MIN_RMS:
        reasons.append("Signal level is too faint.")
        score -= 35
    if model_rms < MIN_MODEL_WINDOW_RMS:
        reasons.append("First 5 seconds contain almost no signal.")
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
    """Compute dominant pitch and Bell vs. Diaphragm stethoscope guidance."""
    if len(model_audio) == 0:
        return {"dom_hz": 0.0, "side": "Bell / Diaphragm", "band_note": "—"}

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
    return {"dom_hz": dom_hz, "side": side, "band_note": band_note}


# ============================================================
# INFERENCE
# ============================================================

def predict_binary(model: Any, X: np.ndarray) -> tuple[str, float, float, float]:
    """Run binary normal/abnormal classification."""
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


def format_label(raw: str) -> str:
    return str(raw).replace("_", " ").title()


def clamp_prob(val: float) -> float:
    return min(max(float(val), 0.0), 1.0)


# ============================================================
# PLOTS (CLEAN & MINIMAL)
# ============================================================

def render_spectrogram_figure(model_audio: np.ndarray) -> plt.Figure:
    """Clean 20–2000 Hz clinical frequency map."""
    stft_mag = np.abs(librosa.stft(model_audio, n_fft=2048, hop_length=256))
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=2048)
    mask = (freqs >= 20) & (freqs <= 2000)
    spec_db = librosa.amplitude_to_db(stft_mag[mask, :], ref=np.max)

    fig, ax = plt.subplots(figsize=(6.8, 3.0), dpi=140)
    fig.patch.set_facecolor("#090d16")
    ax.set_facecolor("#090d16")

    img = ax.imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, MODEL_DURATION, 20, 2000],
    )
    ax.axhline(200, color="#38bdf8", linestyle="--", linewidth=0.8, alpha=0.65)
    ax.axhline(600, color="#34d399", linestyle="--", linewidth=0.8, alpha=0.65)
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


def render_waveform_figure(audio: np.ndarray) -> plt.Figure:
    """Clean waveform + amplitude envelope."""
    fig, ax = plt.subplots(figsize=(6.8, 3.0), dpi=140)
    fig.patch.set_facecolor("#090d16")
    ax.set_facecolor("#090d16")

    dur = len(audio) / SAMPLE_RATE
    t = np.linspace(0, dur, num=len(audio))
    ax.plot(t, audio, color="#38bdf8", linewidth=0.65, alpha=0.85)
    ax.fill_between(t, audio, 0, color="#38bdf8", alpha=0.14)

    env = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
    ax.plot(np.linspace(0, dur, num=len(env)), env * 2.1, color="#fbbf24", linewidth=1.1)

    ax.set_title("Cadence & Envelope", color="#e2e8f0", fontsize=10, pad=8)
    ax.set_xlabel("Time (s)", color="#94a3b8", fontsize=8.5)
    ax.set_xlim(0, max(MODEL_DURATION, dur))
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")

    fig.tight_layout()
    return fig


# ============================================================
# UI SECTIONS
# ============================================================

def render_verdict_card(analysis: dict[str, Any]) -> None:
    """Render clean single-box verdict + split probability bar."""
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
        f'<div class="borderline-pill">Borderline margin ({diff:.1f} pts) — verify clinically</div>'
        if diff < BORDERLINE_DIFF_THRESHOLD * 100
        else ""
    )

    st.markdown(
        dedent(
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
        ).strip(),
        unsafe_allow_html=True,
    )

    if organ == "Lungs" and label == "ABNORMAL" and analysis["lung_type"] is not None:
        with st.container(border=True):
            st.markdown('<div class="field-lbl">Detected Lung Sound Pattern</div>', unsafe_allow_html=True)
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
            st.markdown("".join(bars), unsafe_allow_html=True)


def render_trainee_guide(analysis: dict[str, Any]) -> None:
    """Render concise, high-signal bedside cues for a young doctor."""
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

    st.markdown(
        dedent(
            f"""
            <div class="kpi-row">
            <div class="kpi-tile"><div class="kpi-k">Use Stethoscope</div><div class="kpi-v">{html.escape(prof["side"])}</div></div>
            <div class="kpi-tile"><div class="kpi-k">Dominant Pitch</div><div class="kpi-v">{prof["dom_hz"]:.0f} Hz ({html.escape(prof["band_note"])})</div></div>
            <div class="kpi-tile"><div class="kpi-k">Bedside Focus</div><div class="kpi-v">{html.escape(exam_focus)}</div></div>
            </div>
            <div class="cue-grid">
            <div class="cue-card"><div class="cue-head">Bedside Action</div><ul class="cue-list">{"".join(bedside_cues)}</ul></div>
            <div class="cue-card"><div class="cue-head">Clinical Watch-Out</div><ul class="cue-list">{"".join(watch_out)}</ul></div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )

    if label == "NORMAL":
        cfg = ORGAN_HISTORY_CONFIG[organ]
        st.markdown("---")
        st.markdown(f'<div class="field-lbl">{organ} Medical History Check (Normal Screen)</div>', unsafe_allow_html=True)
        with st.form(f"history_{organ.lower()}"):
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                age = st.number_input("Age", min_value=0, max_value=120, value=25, step=1)
                conditions = st.multiselect(f"Prior {organ.lower()} history", cfg["conditions"])
                exposures = st.multiselect("Exposure", cfg["exposures"])
            with c2:
                symptoms = st.multiselect("Current symptoms", cfg["symptoms"])
                red_flags = st.multiselect("Red flags", cfg["red_flags"])

            submitted = st.form_submit_button("Check Next Step", type="primary", use_container_width=True)

        if submitted:
            if red_flags:
                msg = f"URGENT EVALUATION — {organ} red-flag symptoms override a normal AI clip."
            elif symptoms or conditions or exposures:
                msg = f"CLINICAL REVIEW — Discuss {organ.lower()} history, symptoms, or exposures with a senior clinician."
            else:
                msg = f"ROUTINE CARE — No {organ.lower()} risk factors reported (age {age})."
            st.markdown(f'<div class="rec-pill">{html.escape(msg)}</div>', unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

for k in ("analysis", "audio_id", "last_organ", "last_mode"):
    if k not in st.session_state:
        st.session_state[k] = None


# ============================================================
# SIDEBAR (COLLAPSED BY DEFAULT FOR CLEAN PRESENTATION)
# ============================================================

with st.sidebar:
    st.markdown("### Models")
    for key, name, _, _ in MODEL_SPECS:
        ok = key in models
        cls = "dot-ok" if ok else "dot-err"
        txt = "Ready" if ok else "Missing"
        st.markdown(f'<div class="status-row"><span>{html.escape(name)}</span><span class="{cls}">{txt}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        f"Window: {MODEL_DURATION}s @ {SAMPLE_RATE // 1000} kHz • "
        f"Quality min: {MIN_QUALITY_SCORE}/100 • TF {tf.__version__}"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    dedent(
        """
        <div class="nav-bar">
        <div class="nav-brand">
        <div class="nav-logo">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M2 12h3l2.5-6 3.5 12 3-8 2 2h6"/>
        </svg>
        </div>
        <div>
        <h1 class="nav-title">StethoEdge</h1>
        <p class="nav-sub">AI-assisted cardiac &amp; pulmonary auscultation</p>
        </div>
        </div>
        <div class="nav-tag">Prototype • Not for clinical diagnosis</div>
        </div>
        """
    ).strip(),
    unsafe_allow_html=True,
)


# ============================================================
# MAIN STAGE
# ============================================================

left_col, right_col = st.columns([0.92, 1.55], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown('<div class="field-lbl">Organ</div>', unsafe_allow_html=True)
        organ_choice = st.radio("Organ", ["❤️ Heart", "🫁 Lungs"], horizontal=True, label_visibility="collapsed")
        organ = "Heart" if "Heart" in organ_choice else "Lungs"
        if st.session_state.last_organ != organ:
            st.session_state.last_organ = organ
            st.session_state.analysis = None

        st.markdown('<div class="field-lbl" style="margin-top:0.75rem;">Audio Input</div>', unsafe_allow_html=True)
        mode_choice = st.radio("Mode", ["🎙️ Live", "📁 Upload"], horizontal=True, label_visibility="collapsed")
        input_mode = "Live Recording" if "Live" in mode_choice else "Upload File"
        if st.session_state.last_mode != input_mode:
            st.session_state.last_mode = input_mode
            st.session_state.audio_id = None
            st.session_state.analysis = None

        audio_bytes: bytes | None = None
        audio_suffix = ".wav"
        clip_name = input_mode

        if input_mode == "Live Recording":
            try:
                rec = st.audio_input("Record 5–10s", sample_rate=SAMPLE_RATE)
            except TypeError:
                rec = st.audio_input("Record 5–10s")
            if rec is not None:
                audio_bytes = rec.getvalue()
                clip_name = "Live Mic"
                st.audio(rec)
        else:
            up = st.file_uploader("Upload WAV/MP3/FLAC/OGG", type=["wav", "mp3", "flac", "ogg"], label_visibility="collapsed")
            if up is not None:
                audio_bytes = up.getvalue()
                audio_suffix = Path(up.name).suffix.lower() or ".wav"
                clip_name = up.name
                st.audio(up)

        if audio_bytes is not None:
            cid = hashlib.sha256(audio_bytes).hexdigest()
            if st.session_state.audio_id != cid:
                st.session_state.audio_id = cid
                st.session_state.analysis = None
            run_clicked = st.button("Run Screening", type="primary", use_container_width=True)
        else:
            st.session_state.audio_id = None
            st.session_state.analysis = None
            run_clicked = False

with right_col:
    q_fail: dict[str, Any] | None = None

    if run_clicked and audio_bytes is not None:
        try:
            with st.spinner("Screening audio..."):
                audio, duration = load_audio_bytes(audio_bytes, suffix=audio_suffix)
                quality = assess_audio_quality(audio, duration, input_mode)

                if not quality["acceptable"]:
                    st.session_state.analysis = None
                    q_fail = quality
                else:
                    mkey = "heart" if organ == "Heart" else "lung"
                    if mkey not in models:
                        st.session_state.analysis = None
                        st.error(f"{organ} model missing in `models/`.")
                    else:
                        model_audio, _, X = build_model_input(audio)
                        label, conf, norm_p, abn_p = predict_binary(models[mkey], X)
                        l_type, l_conf, l_probs = (
                            predict_lung_type(X)
                            if organ == "Lungs" and label == "ABNORMAL" and "lung_type" in models and "lung_labels" in models
                            else (None, None, None)
                        )
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
                        }
        except Exception as exc:
            st.session_state.analysis = None
            st.error("Unable to process audio.")
            st.exception(exc)

    analysis = st.session_state.analysis

    if q_fail is not None:
        st.markdown(
            f'<div class="q-fail"><strong>Recording rejected ({q_fail["score"]}/100)</strong> — '
            f'{" ".join(html.escape(r) for r in q_fail["reasons"])}</div>',
            unsafe_allow_html=True,
        )
    elif analysis is not None:
        render_verdict_card(analysis)
        tab_guide, tab_spec = st.tabs(["Clinical Guide", "Audio & Spectrum"])
        with tab_guide:
            render_trainee_guide(analysis)
        with tab_spec:
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                fig_s = render_spectrogram_figure(analysis["model_audio"])
                st.pyplot(fig_s)
                plt.close(fig_s)
            with c2:
                fig_w = render_waveform_figure(analysis["audio"])
                st.pyplot(fig_w)
                plt.close(fig_w)
    else:
        st.markdown(
            '<div class="calm-empty">Select <strong>Heart</strong> or <strong>Lungs</strong> and add a recording on the left to begin.</div>',
            unsafe_allow_html=True,
        )
