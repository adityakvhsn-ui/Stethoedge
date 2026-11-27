"""
StethoEdge - TensorFlow/Keras Audio Screening Demo

Models:
    models/heart_normal_abnormal.keras
    models/lung_normal_abnormal.keras
    models/lung_abnormal_type.keras
    models/lung_abnormal_labels.npy

Run:
    python -m streamlit run demo.py

This is a hackathon/research prototype and is not a clinical diagnostic tool.
"""

from pathlib import Path
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

SAMPLE_RATE = 16000
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

MODEL_SPECS = [
    ("heart", "Heart Screening", HEART_MODEL_PATH, "keras"),
    ("lung", "Lung Screening", LUNG_MODEL_PATH, "keras"),
    ("lung_type", "Lung Sound Classifier", LUNG_TYPE_MODEL_PATH, "keras"),
    ("lung_labels", "Lung Class Labels", LUNG_LABELS_PATH, "npy"),
]


# ============================================================
# PAGE & STYLING
# ============================================================

st.set_page_config(
    page_title="StethoEdge",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1250px;
    }

    .hero-card {
        padding: 1.5rem 1.75rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.12), rgba(16, 185, 129, 0.08));
        border: 1px solid rgba(56, 189, 248, 0.25);
        margin-bottom: 1.25rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.14);
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-bottom: 0.55rem;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        line-height: 1.15;
        margin: 0;
    }

    .hero-subtitle {
        opacity: 0.8;
        font-size: 1rem;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }

    .disclaimer-bar {
        padding: 0.75rem 1rem;
        border-radius: 10px;
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.35);
        font-size: 0.88rem;
        margin-bottom: 1.5rem;
    }

    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.65;
        margin-bottom: 0.35rem;
    }

    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.55rem 0.75rem;
        border-radius: 10px;
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.16);
        margin-bottom: 0.45rem;
        font-size: 0.88rem;
    }

    .pill-ok {
        font-size: 0.74rem;
        font-weight: 700;
        color: #22c55e;
        background: rgba(34, 197, 94, 0.14);
        border: 1px solid rgba(34, 197, 94, 0.35);
        padding: 0.12rem 0.5rem;
        border-radius: 999px;
    }

    .pill-err {
        font-size: 0.74rem;
        font-weight: 700;
        color: #ef4444;
        background: rgba(239, 68, 68, 0.14);
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 0.12rem 0.5rem;
        border-radius: 999px;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1rem;
    }

    .result-normal {
        background: rgba(34, 197, 94, 0.12);
        border: 1.5px solid rgba(34, 197, 94, 0.55);
    }

    .result-abnormal {
        background: rgba(239, 68, 68, 0.12);
        border: 1.5px solid rgba(239, 68, 68, 0.55);
    }

    .result-kicker {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.75;
    }

    .result-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 0.03em;
        margin: 0.2rem 0;
    }

    .result-normal .result-title {
        color: #22c55e;
    }

    .result-abnormal .result-title {
        color: #ef4444;
    }

    .result-confidence {
        font-size: 1rem;
        font-weight: 600;
        opacity: 0.9;
    }

    .placeholder-card {
        padding: 2.5rem 1.75rem;
        border-radius: 16px;
        border: 1px dashed rgba(148, 163, 184, 0.35);
        background: rgba(148, 163, 184, 0.04);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models() -> dict[str, Any]:
    """Load available Keras models and label arrays from disk."""
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
# AUDIO & INFERENCE HELPERS
# ============================================================

def load_and_preprocess_audio(uploaded_file) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an uploaded audio clip, normalize to 5s, and build a Mel spectrogram tensor."""
    suffix = Path(uploaded_file.name).suffix.lower() or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    X = mel_db.astype(np.float32)[np.newaxis, ..., np.newaxis]

    return audio, mel_db, X


def predict_binary(model: Any, X: np.ndarray) -> tuple[str, float, float, float]:
    """Run binary normal/abnormal classification on preprocessed spectrogram input."""
    prediction = model.predict(X, verbose=0)[0]
    normal_prob = float(prediction[0])
    abnormal_prob = float(prediction[1])

    if abnormal_prob >= normal_prob:
        return "ABNORMAL", abnormal_prob, normal_prob, abnormal_prob
    return "NORMAL", normal_prob, normal_prob, abnormal_prob


def predict_lung_type(X: np.ndarray) -> tuple[str, float, np.ndarray]:
    """Classify abnormal lung sound subtype."""
    prediction = models["lung_type"].predict(X, verbose=0)[0]
    labels = models["lung_labels"]
    index = int(np.argmax(prediction))
    return str(labels[index]), float(prediction[index]), prediction


def format_label(raw_label: str) -> str:
    """Format snake_case class labels for display."""
    return str(raw_label).replace("_", " ").title()


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def render_spectrogram_figure(mel_db: np.ndarray) -> plt.Figure:
    """Create a dark-canvas Mel spectrogram figure."""
    fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=140)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    img = ax.imshow(
        mel_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, DURATION, 0, N_MELS],
    )
    ax.set_title("Log-Mel Spectrogram (5.0s Window)", color="#f1f5f9", fontsize=11, pad=10)
    ax.set_xlabel("Time (s)", color="#cbd5e1", fontsize=9.5)
    ax.set_ylabel("Mel Band", color="#cbd5e1", fontsize=9.5)
    ax.tick_params(colors="#94a3b8", labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar.outline.set_edgecolor("#334155")

    fig.tight_layout()
    return fig


def render_waveform_figure(audio: np.ndarray) -> plt.Figure:
    """Create a dark-canvas audio waveform figure."""
    fig, ax = plt.subplots(figsize=(8.5, 3.0), dpi=140)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    time_axis = np.linspace(0, DURATION, num=len(audio))
    ax.plot(time_axis, audio, color="#38bdf8", linewidth=0.75, alpha=0.9)
    ax.axhline(0, color="#334155", linewidth=0.8, linestyle="--")
    ax.set_title("Normalized Audio Waveform (16 kHz)", color="#f1f5f9", fontsize=11, pad=10)
    ax.set_xlabel("Time (s)", color="#cbd5e1", fontsize=9.5)
    ax.set_ylabel("Amplitude", color="#cbd5e1", fontsize=9.5)
    ax.set_xlim(0, DURATION)
    ax.tick_params(colors="#94a3b8", labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    fig.tight_layout()
    return fig


def render_screening_result(
    organ_name: str,
    label: str,
    confidence: float,
    normal_p: float,
    abnormal_p: float,
) -> None:
    """Render the main normal/abnormal verdict card and probability metrics."""
    css_class = "result-normal" if label == "NORMAL" else "result-abnormal"

    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div class="result-kicker">{html.escape(organ_name)} Screening Result</div>
            <div class="result-title">{html.escape(label)}</div>
            <div class="result-confidence">Confidence: {confidence * 100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    col1.metric("Normal Probability", f"{normal_p * 100:.2f}%")
    col2.metric("Abnormal Probability", f"{abnormal_p * 100:.2f}%")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 🩺 Model Registry")

    for key, display_name, _, _ in MODEL_SPECS:
        is_loaded = key in models
        badge_class = "pill-ok" if is_loaded else "pill-err"
        badge_text = "Loaded" if is_loaded else "Missing"
        st.markdown(
            f"""
            <div class="status-row">
                <span>{html.escape(display_name)}</span>
                <span class="{badge_class}">{badge_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### ⚙️ Signal Pipeline")
    st.caption(
        f"• Sample Rate: **{SAMPLE_RATE:,} Hz**\n"
        f"• Window: **{DURATION}s** ({SAMPLES:,} samples)\n"
        f"• Mel Bands: **{N_MELS}** (`n_fft={N_FFT}`, `hop={HOP_LENGTH}`)\n"
        f"• TensorFlow: **{tf.__version__}**"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">Research & Hackathon Prototype</div>
        <h1 class="hero-title">🩺 StethoEdge</h1>
        <p class="hero-subtitle">
            AI-assisted heart and lung auscultation screening powered by Mel-spectrogram CNNs
        </p>
    </div>
    <div class="disclaimer-bar">
        <strong>Notice:</strong> Results are generated by an experimental deep-learning prototype
        and are <strong>not</strong> a clinical diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MAIN WORKSPACE
# ============================================================

left_col, right_col = st.columns([1, 1.35], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown('<div class="section-label">Step 1 • Modality</div>', unsafe_allow_html=True)
        screening_choice = st.radio(
            "Select auscultation target",
            ["❤️ Heart", "🫁 Lungs"],
            horizontal=True,
            label_visibility="collapsed",
        )
        screening_type = "Heart" if "Heart" in screening_choice else "Lungs"

        st.markdown("---")
        st.markdown('<div class="section-label">Step 2 • Audio Recording</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload heart or lung audio",
            type=["wav", "mp3", "flac", "ogg"],
            label_visibility="collapsed",
        )
        st.caption(
            "WAV is recommended. The first **5 seconds** are analyzed; shorter clips are zero-padded."
        )

        if uploaded_file is not None:
            st.audio(uploaded_file)
            run_clicked = st.button(
                "Run AI Screening",
                type="primary",
                use_container_width=True,
            )
        else:
            run_clicked = False

with right_col:
    if uploaded_file is None:
        st.markdown(
            """
            <div class="placeholder-card">
                <h3 style="margin-top:0;">Ready for Auscultation Screening</h3>
                <p style="opacity:0.75; margin-bottom:0;">
                    Select <strong>Heart</strong> or <strong>Lungs</strong> and upload an audio recording on the left
                    to view screening predictions, confidence scores, and spectral analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif not run_clicked:
        st.markdown(
            """
            <div class="placeholder-card">
                <h3 style="margin-top:0;">Recording Loaded</h3>
                <p style="opacity:0.75; margin-bottom:0;">
                    Click <strong>Run AI Screening</strong> to process the 5-second audio window
                    and generate predictions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        model_key = "heart" if screening_type == "Heart" else "lung"

        if model_key not in models:
            st.error(f"{screening_type} screening model is missing from `models/`.")
        else:
            try:
                with st.spinner("Extracting Mel spectrogram and running inference..."):
                    audio, mel_db, X = load_and_preprocess_audio(uploaded_file)
                    label, confidence, normal_p, abnormal_p = predict_binary(
                        models[model_key], X
                    )

                with st.container(border=True):
                    render_screening_result(
                        organ_name=screening_type,
                        label=label,
                        confidence=confidence,
                        normal_p=normal_p,
                        abnormal_p=abnormal_p,
                    )

                    if screening_type == "Lungs" and label == "ABNORMAL":
                        st.markdown("---")
                        st.markdown("#### Abnormal Sound Classification")

                        if "lung_type" in models and "lung_labels" in models:
                            type_label, type_conf, type_probs = predict_lung_type(X)
                            st.info(
                                f"Primary detected pattern: **{format_label(type_label)}** "
                                f"({type_conf * 100:.2f}% confidence)"
                            )

                            labels = models["lung_labels"]
                            ranked = sorted(
                                zip(labels, type_probs),
                                key=lambda item: float(item[1]),
                                reverse=True,
                            )
                            for name, prob in ranked:
                                clamped = min(max(float(prob), 0.0), 1.0)
                                st.progress(
                                    clamped,
                                    text=f"{format_label(name)} — {clamped * 100:.2f}%",
                                )
                        else:
                            st.warning("Abnormal-type classifier or labels are unavailable.")

                with st.container(border=True):
                    st.markdown("#### Audio & Spectral Inspection")
                    tab_spec, tab_wave = st.tabs(["Mel Spectrogram", "Waveform"])

                    with tab_spec:
                        fig_spec = render_spectrogram_figure(mel_db)
                        st.pyplot(fig_spec)
                        plt.close(fig_spec)

                    with tab_wave:
                        fig_wave = render_waveform_figure(audio)
                        st.pyplot(fig_wave)
                        plt.close(fig_wave)

                    st.caption(
                        "The 64-band log-Mel spectrogram shown above is the exact input tensor supplied to the neural network."
                    )

            except Exception as exc:
                st.error("Could not process this audio recording.")
                st.exception(exc)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("StethoEdge • TensorFlow/Keras Prototype • Not Clinically Validated")