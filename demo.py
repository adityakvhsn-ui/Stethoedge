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
import hashlib
import html
import os
import tempfile
import textwrap
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
RECORDING_DURATION = 10          # requested live recording target
MODEL_DURATION = 5               # existing models were trained on 5-second windows
MODEL_SAMPLES = SAMPLE_RATE * MODEL_DURATION

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

# Optional About-page details. Fill these in before presenting the project.
# Leave them blank to hide the Team section.
TEAM_NAME = ""
TEAM_MEMBERS: list[str] = []
PROJECT_URL = ""

MODEL_SPECS = [
    ("heart", "Heart Screening", HEART_MODEL_PATH, "keras"),
    ("lung", "Lung Screening", LUNG_MODEL_PATH, "keras"),
    ("lung_type", "Lung Sound Classifier", LUNG_TYPE_MODEL_PATH, "keras"),
    ("lung_labels", "Lung Class Labels", LUNG_LABELS_PATH, "npy"),
]


def html_block(markup: str) -> None:
    """Render an HTML fragment after removing indentation that can create code blocks."""
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)


def segmented_choice(label: str, options: list[str], default: str, key: str) -> str:
    """Use Streamlit's segmented buttons, with a radio fallback for older versions."""
    if hasattr(st, "segmented_control"):
        selected = st.segmented_control(
            label,
            options,
            default=default,
            key=key,
            label_visibility="collapsed",
        )
        return selected or default
    return st.radio(
        label,
        options,
        index=options.index(default),
        horizontal=True,
        key=key,
        label_visibility="collapsed",
    )


# ============================================================
# PAGE & STYLING
# ============================================================

st.set_page_config(
    page_title="StethoEdge",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

html_block(
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

    .quality-pass {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background: rgba(34, 197, 94, 0.10);
        border: 1px solid rgba(34, 197, 94, 0.35);
    }

    .quality-fail {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background: rgba(239, 68, 68, 0.10);
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    .recommendation {
        padding: 1rem 1.1rem;
        border-radius: 12px;
        background: rgba(59, 130, 246, 0.09);
        border: 1px solid rgba(59, 130, 246, 0.30);
    }

    .placeholder-card {
        padding: 2.5rem 1.75rem;
        border-radius: 16px;
        border: 1px dashed rgba(148, 163, 184, 0.35);
        background: rgba(148, 163, 184, 0.04);
        text-align: center;
    }

    .tips-card {
        margin-top: 1rem;
        padding: 1rem 1.1rem;
        border-radius: 12px;
        background: rgba(56, 189, 248, 0.07);
        border: 1px solid rgba(56, 189, 248, 0.22);
        font-size: 0.9rem;
    }
    .tips-card ul { margin: 0.5rem 0 0 1.1rem; padding: 0; }
    .tips-card li { margin: 0.25rem 0; }
    .landing-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 1.4rem 0;
        text-align: left;
    }
    .landing-grid div {
        padding: 0.85rem;
        border-radius: 10px;
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.18);
        font-size: 0.88rem;
    }
    .landing-grid strong { color: #38bdf8; font-size: 1.1rem; }
    .landing-note { opacity: 0.75; font-size: 0.9rem; }
    @media (max-width: 760px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .hero-title { font-size: 1.8rem; }
        .landing-grid { grid-template-columns: 1fr; }
    }
    </style>
    """
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models() -> tuple[dict[str, Any], dict[str, str]]:
    """Load each available model independently; report failures without crashing the app."""
    loaded: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for key, display_name, model_path, kind in MODEL_SPECS:
        if not model_path.exists():
            errors[key] = f"Missing file: {model_path.name}"
            continue

        try:
            if kind == "keras":
                loaded[key] = tf.keras.models.load_model(model_path)
            elif kind == "npy":
                loaded[key] = np.load(model_path, allow_pickle=False)
        except Exception as exc:
            errors[key] = f"Could not load {display_name}: {type(exc).__name__}: {exc}"

    return loaded, errors


models, model_load_errors = load_models()


def render_model_registry() -> None:
    """Show model availability and any loading errors."""
    for key, display_name, model_path, _kind in MODEL_SPECS:
        if key in models:
            status = "Loaded"
            css_class = "pill-ok"
        elif key in model_load_errors and model_path.exists():
            status = "Failed"
            css_class = "pill-err"
        else:
            status = "Missing"
            css_class = "pill-err"

        html_block(
            f"""
            <div class="status-row">
                <span>{html.escape(display_name)}</span>
                <span class="{css_class}">{status}</span>
            </div>
            """
        )
        if key in model_load_errors:
            st.caption(model_load_errors[key])



# ============================================================
# AUDIO HELPERS
# ============================================================

def load_audio_bytes(audio_bytes: bytes, suffix: str = ".wav") -> tuple[np.ndarray, float]:
    """Decode recorded/uploaded audio at 16 kHz mono."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    duration = len(audio) / SAMPLE_RATE
    return audio.astype(np.float32), duration


def build_model_input(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep the existing model interface:
    first 5 seconds -> 64-bin log-Mel spectrogram -> [1, mel, time, 1].
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


def assess_audio_quality(audio: np.ndarray, duration: float) -> dict[str, Any]:
    """
    Technical recording-quality gate.
    This is not a medical-quality assessment.
    """
    if len(audio) == 0:
        return {
            "acceptable": False,
            "score": 0,
            "reasons": ["No audio signal was captured."],
        }

    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    peak = float(np.max(np.abs(audio)))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.99))

    threshold = max(0.005, peak * 0.05)
    silence_ratio = float(np.mean(np.abs(audio) < threshold))

    reasons: list[str] = []
    score = 100

    if duration < 8:
        reasons.append("Recording is shorter than 8 seconds.")
        score -= 35
    elif duration < 9.5:
        score -= 10

    if rms < 0.003:
        reasons.append("Signal level is too weak.")
        score -= 35

    if peak < 0.03:
        reasons.append("Peak amplitude is too low.")
        score -= 15

    if silence_ratio > 0.97:
        reasons.append("Too much silence was detected.")
        score -= 25

    if clipping_ratio > 0.02:
        reasons.append("The recording contains excessive clipping/distortion.")
        score -= 20

    score = max(0, min(100, score))

    # A technical gate should be conservative without being unnecessarily strict.
    acceptable = (
        duration >= 8
        and rms >= 0.003
        and peak >= 0.03
        and silence_ratio <= 0.97
        and clipping_ratio <= 0.02
    )

    if acceptable and not reasons:
        reasons.append("Signal level, duration, silence, and clipping checks passed.")

    return {
        "acceptable": acceptable,
        "score": score,
        "rms": rms,
        "peak": peak,
        "silence_ratio": silence_ratio,
        "clipping_ratio": clipping_ratio,
        "reasons": reasons,
    }


# ============================================================
# INFERENCE HELPERS
# ============================================================

def predict_binary(
    model: Any,
    X: np.ndarray,
) -> tuple[str, float, float, float]:
    """Run binary normal/abnormal classification."""
    prediction = np.asarray(model.predict(X, verbose=0)[0]).reshape(-1)

    if prediction.size < 2:
        # Supports a sigmoid-style one-output model as well.
        abnormal_prob = float(prediction[0])
        normal_prob = 1.0 - abnormal_prob
    else:
        normal_prob = float(prediction[0])
        abnormal_prob = float(prediction[1])

    if abnormal_prob >= normal_prob:
        return "ABNORMAL", abnormal_prob, normal_prob, abnormal_prob

    return "NORMAL", normal_prob, normal_prob, abnormal_prob


def predict_lung_type(
    X: np.ndarray,
) -> tuple[str, float, np.ndarray]:
    """Classify abnormal lung sound subtype."""
    prediction = np.asarray(models["lung_type"].predict(X, verbose=0)[0]).reshape(-1)
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
        extent=[0, MODEL_DURATION, 0, N_MELS],
    )

    ax.set_title(
        "Log-Mel Spectrogram (5.0s Model Window)",
        color="#f1f5f9",
        fontsize=11,
        pad=10,
    )
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

    time_axis = np.linspace(0, len(audio) / SAMPLE_RATE, num=len(audio))
    ax.plot(time_axis, audio, color="#38bdf8", linewidth=0.75, alpha=0.9)
    ax.axhline(0, color="#334155", linewidth=0.8, linestyle="--")

    ax.set_title(
        "Recorded Audio Waveform (16 kHz)",
        color="#f1f5f9",
        fontsize=11,
        pad=10,
    )
    ax.set_xlabel("Time (s)", color="#cbd5e1", fontsize=9.5)
    ax.set_ylabel("Amplitude", color="#cbd5e1", fontsize=9.5)
    ax.set_xlim(0, max(RECORDING_DURATION, len(audio) / SAMPLE_RATE))
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

    html_block(
        f"""
        <div class="result-card {css_class}">
            <div class="result-kicker">{html.escape(organ_name)} Screening Result</div>
            <div class="result-title">✓ Prediction: {html.escape(label)}</div>
            <div class="result-confidence">Model output: {confidence * 100:.0f}%</div>
        </div>
        """
    )

    col1, col2 = st.columns(2)
    col1.metric("Normal Probability", f"{normal_p * 100:.1f}%")
    col2.metric("Abnormal Probability", f"{abnormal_p * 100:.1f}%")


# ============================================================
# ORGAN-SPECIFIC HISTORY / NEXT-STEP GUIDANCE
# ============================================================

def render_patient_history_and_recommendation(
    organ: str,
    normal_p: float,
    abnormal_p: float,
) -> None:
    """Collect organ-specific context when model probabilities are close.

    The score is a transparent prototype rule, not a validated clinical risk
    score or a probability of disease. Urgent symptoms always override it.
    """
    probability_gap = abs(normal_p - abnormal_p) * 100.0

    if probability_gap > 10.0:
        return

    is_heart = organ.lower() == "heart"
    organ_label = "Heart" if is_heart else "Lung"

    st.markdown("---")
    st.markdown(f"### {organ_label} History & Consultation Guidance")
    st.caption(
        f"The model probabilities are close (difference: {probability_gap:.1f} "
        "percentage points). Please answer the relevant questions. The score "
        "helps organise the history; it does not diagnose a condition."
    )

    with st.form(f"patient_history_{organ.lower()}_form"):
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=25,
            step=1,
            key=f"{organ.lower()}_history_age",
        )

        if is_heart:
            st.markdown("#### Previous heart history")
            history = st.multiselect(
                "Have you ever been diagnosed with any of these?",
                [
                    "Coronary artery disease / angina",
                    "Previous heart attack or heart procedure",
                    "Heart failure",
                    "Irregular heartbeat / arrhythmia",
                    "High blood pressure",
                    "Diabetes",
                    "High cholesterol",
                    "Previous stroke / TIA",
                ],
                key="heart_history_conditions",
            )
            family_history = st.checkbox(
                "A parent or sibling had heart disease at a relatively young age",
                key="heart_family_history",
            )
            tobacco = st.radio(
                "Smoking / tobacco history",
                ["Never", "Current", "Former"],
                horizontal=True,
                key="heart_tobacco_history",
            )

            st.markdown("#### Current heart-related symptoms")
            symptoms = st.multiselect(
                "Are you currently experiencing any of these?",
                [
                    "Chest discomfort or pressure (not severe or ongoing)",
                    "Shortness of breath during activity",
                    "Palpitations / racing heartbeat",
                    "Dizziness or light-headedness",
                    "Unexplained swelling of the ankles or legs",
                    "Unusual fatigue",
                ],
                key="heart_history_symptoms",
            )
            urgent = st.multiselect(
                "Urgent warning symptoms — select any that apply",
                [
                    "Severe, persistent, or new chest pain/pressure now",
                    "Severe shortness of breath at rest",
                    "Fainting or near-fainting now",
                    "Sudden face drooping, arm weakness, or trouble speaking",
                ],
                help="These symptoms need urgent medical assessment regardless of the model result or score.",
                key="heart_history_urgent",
            )

            history_points = {
                "Coronary artery disease / angina": 20,
                "Previous heart attack or heart procedure": 22,
                "Heart failure": 25,
                "Irregular heartbeat / arrhythmia": 12,
                "High blood pressure": 8,
                "Diabetes": 8,
                "High cholesterol": 6,
                "Previous stroke / TIA": 10,
            }
            symptom_points = {
                "Chest discomfort or pressure (not severe or ongoing)": 18,
                "Shortness of breath during activity": 15,
                "Palpitations / racing heartbeat": 10,
                "Dizziness or light-headedness": 8,
                "Unexplained swelling of the ankles or legs": 12,
                "Unusual fatigue": 6,
            }
            score_raw = sum(history_points[x] for x in history)
            score_raw += sum(symptom_points[x] for x in symptoms)
            score_raw += 8 if family_history else 0
            score_raw += {"Never": 0, "Current": 10, "Former": 5}[tobacco]

        else:
            st.markdown("#### Previous lung / respiratory history")
            history = st.multiselect(
                "Have you ever been diagnosed with any of these?",
                [
                    "Asthma",
                    "COPD / emphysema",
                    "Bronchiectasis",
                    "Interstitial lung disease / pulmonary fibrosis",
                    "Previous tuberculosis (TB)",
                    "Repeated pneumonia or serious chest infections",
                ],
                key="lung_history_conditions",
            )
            exposures = st.multiselect(
                "Smoking and environmental exposure history",
                [
                    "Current cigarette / bidi smoking",
                    "Former cigarette / bidi smoking",
                    "Vaping",
                    "Regular second-hand smoke exposure",
                    "Work-related dust, fumes, or chemicals",
                    "Regular indoor biomass / wood / cooking smoke",
                ],
                key="lung_history_exposures",
            )

            st.markdown("#### Current respiratory symptoms")
            symptoms = st.multiselect(
                "Are you currently experiencing any of these?",
                [
                    "Persistent cough (around 3 weeks or more)",
                    "Cough with sputum / phlegm",
                    "Wheezing",
                    "Shortness of breath while walking or being active",
                    "Repeated chest infections",
                    "Fever with a current respiratory illness",
                    "Chest tightness or discomfort",
                ],
                key="lung_history_symptoms",
            )
            urgent = st.multiselect(
                "Urgent warning symptoms — select any that apply",
                [
                    "Severe difficulty breathing at rest",
                    "Blue or grey lips / face",
                    "Coughing up blood",
                    "New confusion or fainting with breathing symptoms",
                    "Chest pain together with breathing difficulty",
                ],
                help="These symptoms need urgent medical assessment regardless of the model result or score.",
                key="lung_history_urgent",
            )

            history_points = {
                "Asthma": 12,
                "COPD / emphysema": 22,
                "Bronchiectasis": 20,
                "Interstitial lung disease / pulmonary fibrosis": 22,
                "Previous tuberculosis (TB)": 15,
                "Repeated pneumonia or serious chest infections": 10,
            }
            exposure_points = {
                "Current cigarette / bidi smoking": 12,
                "Former cigarette / bidi smoking": 6,
                "Vaping": 10,
                "Regular second-hand smoke exposure": 5,
                "Work-related dust, fumes, or chemicals": 8,
                "Regular indoor biomass / wood / cooking smoke": 8,
            }
            symptom_points = {
                "Persistent cough (around 3 weeks or more)": 15,
                "Cough with sputum / phlegm": 8,
                "Wheezing": 12,
                "Shortness of breath while walking or being active": 15,
                "Repeated chest infections": 10,
                "Fever with a current respiratory illness": 5,
                "Chest tightness or discomfort": 8,
            }
            score_raw = sum(history_points[x] for x in history)
            score_raw += sum(exposure_points[x] for x in exposures)
            score_raw += sum(symptom_points[x] for x in symptoms)

        submitted = st.form_submit_button(
            "Calculate History Score",
            type="primary",
            use_container_width=True,
        )

    result_key = (
        f"{organ.lower()}_history_result_"
        f"{st.session_state.get('active_audio_identity', 'current')}"
    )

    if submitted:
        score = min(100, int(score_raw))

        if urgent:
            result_level = "URGENT MEDICAL ASSESSMENT"
            result_message = (
                "You selected a warning symptom. Do not rely on the AI result "
                "or this score. Seek urgent medical care now. For severe "
                "breathing difficulty, ongoing severe chest pain, blue/grey "
                "lips, fainting, or stroke-like symptoms, contact local "
                "emergency services."
            )
        elif score >= 60:
            result_level = "PROMPT DOCTOR REVIEW RECOMMENDED"
            result_message = (
                "The history answers include several factors that should be "
                "reviewed by a qualified clinician. Arrange a medical "
                "assessment promptly; do not use the AI result to delay care."
            )
        elif score >= 40:
            result_level = "DOCTOR CONSULTATION RECOMMENDED"
            result_message = (
                "Arrange a clinical appointment to discuss the symptoms and "
                "medical history, especially if they are persistent or worsening."
            )
        elif score >= 20:
            result_level = "MILD / NON-URGENT CONSULTATION TO CONSIDER"
            result_message = (
                "Discuss the history with a healthcare professional if symptoms "
                "persist, recur, or affect daily activities."
            )
        else:
            result_level = "LOWER HISTORY CONTEXT SCORE"
            result_message = (
                "This does not rule out a medical condition. Follow routine "
                "care and seek medical advice if symptoms develop, persist, "
                "or worsen."
            )

        st.session_state[result_key] = {
            "score": score,
            "age": age,
            "urgent": bool(urgent),
            "level": result_level,
            "message": result_message,
        }

    result = st.session_state.get(result_key)
    if not result:
        return

    st.markdown("#### History context score")
    st.metric("Context score (not a disease probability)", f"{result['score']}/100")
    st.progress(result["score"])

    if result["urgent"]:
        st.error(f"{result['level']}: {result['message']}")
        st.caption("Urgent symptoms override the numerical score.")
    elif result["score"] >= 60:
        st.error(f"{result['level']}. {result['message']}")
    elif result["score"] >= 40:
        st.warning(f"{result['level']}. {result['message']}")
    elif result["score"] >= 20:
        st.info(f"{result['level']}. {result['message']}")
    else:
        st.success(f"{result['level']}. {result['message']}")

    st.caption(
        f"Age recorded: {result['age']}. This is an experimental, rule-based "
        "score created for the prototype; it is not clinically validated and "
        "must not be interpreted as a diagnosis or a probability of disease. "
        "The audio model and questionnaire can both be wrong."
    )


# ============================================================
# NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## 🩺 StethoEdge")
    st.caption("Cardiopulmonary audio screening prototype")
    selected_page = st.radio(
        "Navigation",
        ["Screening", "How to use", "About"],
        index=0,
        key="app_navigation",
    )
    st.markdown("---")
    st.caption("Research prototype • Not clinically validated")


if selected_page == "Screening":
    # ============================================================
    # HEADER
    # ============================================================

    html_block(
        """
        <div class="hero-card">
            <div class="hero-badge">Research & Hackathon Prototype</div>
            <h1 class="hero-title">🩺 StethoEdge</h1>
            <p class="hero-subtitle">
                AI-assisted heart and lung auscultation screening powered by Mel-spectrogram CNNs
            </p>
        </div>

        <div class="disclaimer-bar">
            <strong>Notice:</strong> Results are generated by an experimental deep-learning
            prototype and are <strong>not a clinical diagnosis or medical clearance</strong>.
        </div>
        """,
    )


    # ============================================================
    # MAIN WORKSPACE
    # ============================================================

    left_col, right_col = st.columns([1, 1.35], gap="large")

    with left_col:
        with st.container(border=True):
            html_block('<div class="section-label">Step 1 • Modality</div>')

            screening_choice = segmented_choice(
                "Select auscultation target",
                ["❤️ Heart", "🫁 Lungs"],
                default="❤️ Heart",
                key="screening_modality",
            )
            screening_type = "Heart" if "Heart" in screening_choice else "Lungs"

            st.markdown("---")
            html_block('<div class="section-label">Step 2 • Audio Input</div>')

            audio_source = segmented_choice(
                "Choose how to provide audio",
                ["Live recording", "Upload audio file"],
                default="Live recording",
                key="audio_source_mode",
            )

            selected_audio_bytes = None
            selected_audio_name = "recording.wav"
            run_clicked = False

            if audio_source == "Live recording":
                st.markdown(
                    f"**LIVE AUSCULTATION**  \n"
                    f"Recording target: **{RECORDING_DURATION} seconds**  \n"
                    f"Type: **{screening_type}**"
                )

                recorded_audio = st.audio_input(
                    "Start live recording",
                    sample_rate=SAMPLE_RATE,
                    key="live_audio_input",
                )

                if recorded_audio is not None:
                    selected_audio_bytes = recorded_audio.getvalue()
                    selected_audio_name = "recording.wav"
                    st.audio(selected_audio_bytes, format="audio/wav")
                    run_clicked = st.button(
                        "Analyze Auscultation",
                        type="primary",
                        use_container_width=True,
                        key="analyze_live_audio",
                    )

                st.caption(
                    "Record close to the stethoscope/microphone. The technical quality "
                    "check runs before inference. The models analyze the first 5 seconds."
                )

            else:
                uploaded_audio = st.file_uploader(
                    "Upload heart or lung audio",
                    type=["wav", "mp3", "flac", "ogg"],
                    key="uploaded_audio_file",
                    help="WAV is recommended. Upload a recording of at least 8 seconds for the quality check.",
                )

                if uploaded_audio is not None:
                    selected_audio_bytes = uploaded_audio.getvalue()
                    selected_audio_name = uploaded_audio.name
                    audio_suffix = Path(selected_audio_name).suffix.lower()
                    audio_mime = {
                        ".wav": "audio/wav",
                        ".mp3": "audio/mpeg",
                        ".flac": "audio/flac",
                        ".ogg": "audio/ogg",
                    }.get(audio_suffix, "audio/wav")
                    st.audio(selected_audio_bytes, format=audio_mime)
                    run_clicked = st.button(
                        "Run AI Screening",
                        type="primary",
                        use_container_width=True,
                        key="analyze_uploaded_audio",
                    )

                st.caption(
                    "WAV is recommended. The technical quality check is applied to "
                    "uploaded recordings too. The model analyzes the first 5 seconds."
                )

            html_block(
                """
                <div class="tips-card">
                    <strong>Recording tips</strong>
                    <ul>
                        <li>Use a quiet room and reduce background conversation.</li>
                        <li>Keep the microphone or stethoscope steady against the intended location.</li>
                        <li>Capture around 10 seconds; the current model uses the first 5 seconds.</li>
                    </ul>
                </div>
                """
            )

            # Clear stale results whenever the selected audio, modality, or source changes.
            if selected_audio_bytes is not None:
                audio_identity = hashlib.sha256(
                    selected_audio_bytes
                    + screening_type.encode("utf-8")
                    + audio_source.encode("utf-8")
                    + selected_audio_name.encode("utf-8")
                ).hexdigest()
                if st.session_state.get("active_audio_identity") != audio_identity:
                    st.session_state.active_audio_identity = audio_identity
                    st.session_state.analysis = None
                    # Do not carry one patient's questionnaire answers to a new audio sample.
                    for history_key in (
                        "heart_history_age",
                        "heart_history_conditions",
                        "heart_family_history",
                        "heart_tobacco_history",
                        "heart_history_symptoms",
                        "heart_history_urgent",
                        "lung_history_age",
                        "lung_history_conditions",
                        "lung_history_exposures",
                        "lung_history_symptoms",
                        "lung_history_urgent",
                    ):
                        st.session_state.pop(history_key, None)
            else:
                st.session_state.active_audio_identity = None
                st.session_state.analysis = None

    with right_col:
        analysis = st.session_state.get("analysis")

        if selected_audio_bytes is None and analysis is None:
            html_block(
                """
                <div class="placeholder-card">
                    <h3 style="margin-top:0;">Ready for Auscultation</h3>
                    <p style="opacity:0.75;">
                        Choose an organ, record or upload a sound sample, then review the
                        model output and audio features.
                    </p>
                    <div class="landing-grid">
                        <div><strong>01</strong><br>Choose heart or lungs</div>
                        <div><strong>02</strong><br>Record or upload audio</div>
                        <div><strong>03</strong><br>Review screening results</div>
                    </div>
                    <p class="landing-note">
                        Results include normal/abnormal model outputs, a recording-quality
                        check, and waveform / Mel-spectrogram inspection. Lung subtype output
                        is shown when the binary lung model predicts abnormal.
                    </p>
                </div>
                """
            )

        elif run_clicked and selected_audio_bytes is not None:
            try:
                with st.spinner("Analyzing..."):
                    audio_bytes = selected_audio_bytes
                    audio_suffix = Path(selected_audio_name).suffix.lower() or ".wav"
                    audio, duration = load_audio_bytes(audio_bytes, suffix=audio_suffix)
                    quality = assess_audio_quality(audio, duration)

                    if not quality["acceptable"]:
                        st.session_state.analysis = None

                        html_block(
                            f"""
                            <div class="quality-fail">
                                <strong>✗ Recording quality: NOT ACCEPTABLE</strong><br>
                                Technical quality score: {quality["score"]}/100
                            </div>
                            """
                        )

                        st.markdown("#### Why the recording was rejected")

                        for reason in quality["reasons"]:
                            st.write(f"• {reason}")

                        st.warning(
                            "Record again with the microphone/stethoscope positioned "
                            "correctly and with minimal background noise."
                        )
                        st.stop()

                    model_audio, mel_db, X = build_model_input(audio)

                    model_key = "heart" if screening_type == "Heart" else "lung"

                    if model_key not in models:
                        st.error(
                            f"{screening_type} screening model is missing from `models/`."
                        )
                        st.stop()

                    label, confidence, normal_p, abnormal_p = predict_binary(
                        models[model_key],
                        X,
                    )

                    lung_type = None
                    lung_type_conf = None
                    lung_type_probs = None

                    if (
                        screening_type == "Lungs"
                        and label == "ABNORMAL"
                        and "lung_type" in models
                        and "lung_labels" in models
                    ):
                        lung_type, lung_type_conf, lung_type_probs = predict_lung_type(X)

                    st.session_state.analysis = {
                        "screening_type": screening_type,
                        "audio_name": selected_audio_name,
                        "audio_source": audio_source,
                        "duration": duration,
                        "quality": quality,
                        "audio": audio,
                        "model_audio": model_audio,
                        "mel_db": mel_db,
                        "label": label,
                        "confidence": confidence,
                        "normal_p": normal_p,
                        "abnormal_p": abnormal_p,
                        "lung_type": lung_type,
                        "lung_type_conf": lung_type_conf,
                        "lung_type_probs": lung_type_probs,
                    }

                    st.session_state.analysis_id = st.session_state.active_audio_identity

                st.rerun()

            except Exception as exc:
                st.error("Could not process this audio recording.")
                st.exception(exc)

        elif selected_audio_bytes is not None and not run_clicked and analysis is None:
            html_block(
                """
                <div class="placeholder-card">
                    <h3 style="margin-top:0;">Audio Ready</h3>
                    <p style="opacity:0.75; margin-bottom:0;">
                        Click the screening button to check the recording quality,
                        run the selected model, and inspect the audio features.
                    </p>
                </div>
                """
            )

        elif analysis is not None:
            # ----------------------------------------------------
            # RESULTS
            # ----------------------------------------------------
            duration = analysis["duration"]
            quality = analysis["quality"]
            label = analysis["label"]
            confidence = analysis["confidence"]
            normal_p = analysis["normal_p"]
            abnormal_p = analysis["abnormal_p"]

            st.markdown("### AUSCULTATION ANALYSIS")
            st.write(
                f"**Audio:** {analysis.get('audio_name', 'recording.wav')}  \n"
                f"**Duration:** {duration:.1f} seconds  \n"
                f"**Input:** {analysis.get('audio_source', 'Audio')}  \n"
                f"**Type:** {analysis['screening_type']}"
            )

            html_block(
                f"""
                <div class="quality-pass">
                    <strong>✓ Recording quality: ACCEPTABLE</strong><br>
                    Technical quality score: {quality["score"]}/100
                </div>
                """
            )

            st.markdown("")

            render_screening_result(
                organ_name=analysis["screening_type"],
                label=label,
                confidence=confidence,
                normal_p=normal_p,
                abnormal_p=abnormal_p,
            )

            if analysis["screening_type"] == "Lungs" and label == "ABNORMAL":
                st.markdown("---")
                st.markdown("#### Abnormal Sound Classification")

                if analysis["lung_type"] is not None:
                    st.info(
                        f"Primary detected pattern: **"
                        f"{format_label(analysis['lung_type'])}** "
                        f"({analysis['lung_type_conf'] * 100:.1f}% confidence)"
                    )

                    labels = models["lung_labels"]
                    probs = analysis["lung_type_probs"]

                    ranked = sorted(
                        zip(labels, probs),
                        key=lambda item: float(item[1]),
                        reverse=True,
                    )

                    for name, prob in ranked:
                        clamped = min(max(float(prob), 0.0), 1.0)
                        st.progress(
                            clamped,
                            text=f"{format_label(name)} — {clamped * 100:.1f}%",
                        )

            # ----------------------------------------------------
            # AUDIO + SPECTRAL INSPECTION
            # ----------------------------------------------------
            with st.container(border=True):
                st.markdown("#### Audio & Spectral Inspection")

                tab_spec, tab_wave = st.tabs(
                    ["Mel Spectrogram", "Waveform"]
                )

                with tab_spec:
                    fig_spec = render_spectrogram_figure(analysis["mel_db"])
                    st.pyplot(fig_spec)
                    plt.close(fig_spec)

                    st.caption(
                        "This 64-band log-Mel spectrogram is the same 5-second "
                        "model input representation used for inference."
                    )

                with tab_wave:
                    fig_wave = render_waveform_figure(analysis["audio"])
                    st.pyplot(fig_wave)
                    plt.close(fig_wave)

            # ----------------------------------------------------
            # ORGAN-SPECIFIC HISTORY WHEN THE MODEL IS UNCERTAIN
            # ----------------------------------------------------
            probability_gap = abs(normal_p - abnormal_p) * 100.0

            if probability_gap <= 10.0:
                render_patient_history_and_recommendation(
                    analysis["screening_type"],
                    normal_p,
                    abnormal_p,
                )
            elif label == "ABNORMAL":
                st.markdown("---")
                st.markdown("### Next Step")
                st.warning(
                    "The screening model detected an abnormal sound pattern. "
                    "Arrange review with a qualified healthcare professional. "
                    "This model result is not a diagnosis."
                )
                st.caption(
                    "Seek care based on symptoms or concerns even if the model "
                    "probabilities are not close. Urgent symptoms need urgent care."
                )
            else:
                st.markdown("---")
                st.caption(
                    "The model probabilities differ by more than 10 percentage "
                    "points, so the additional history questionnaire is not shown. "
                    "A NORMAL screening result does not rule out disease. Seek "
                    "medical advice for persistent, worsening, or concerning symptoms."
                )


    # ============================================================
    # FOOTER
    # ============================================================

    st.markdown("---")
    st.caption(
        "StethoEdge • TensorFlow/Keras Prototype • Not Clinically Validated"
    )

elif selected_page == "How to use":
    html_block(
        """
        <div class="hero-card">
            <div class="hero-badge">User guide</div>
            <h1 class="hero-title">How to use StethoEdge</h1>
            <p class="hero-subtitle">A short guide to recording, screening and reading the prototype output.</p>
        </div>
        """
    )

    st.markdown("### Screening workflow")
    guide_cols = st.columns(4)
    guide_items = [
        ("01 · Select", "Choose Heart or Lungs on the Screening page."),
        ("02 · Capture", "Use Live recording or upload a WAV, MP3, FLAC or OGG file."),
        ("03 · Check", "Run the technical audio-quality check and model inference."),
        ("04 · Review", "Read the probabilities, subtype output (for abnormal lung results), waveform and Mel spectrogram."),
    ]
    for col, (heading, body) in zip(guide_cols, guide_items):
        with col:
            with st.container(border=True):
                st.markdown(f"**{heading}**")
                st.write(body)

    st.markdown("### Recording and placement tips")
    tips_left, tips_right = st.columns(2)
    with tips_left:
        with st.container(border=True):
            st.markdown("**Before recording**")
            st.write(
                "Use a quiet room, keep the phone or microphone stable, and avoid "
                "handling noise. Select the correct organ before recording."
            )
    with tips_right:
        with st.container(border=True):
            st.markdown("**During recording**")
            st.write(
                "Aim for approximately 10 seconds. Place the stethoscope or microphone "
                "over the intended listening point and avoid rubbing or moving it."
            )

    st.markdown("### Understanding the result")
    st.write(
        "The binary model returns Normal and Abnormal probabilities. The displayed "
        "label is whichever output is larger; these probabilities are model outputs, "
        "not measured disease probabilities. The current model input is the first "
        "5 seconds resampled to 16 kHz mono and converted to a 64-band log-Mel "
        "spectrogram."
    )
    st.write(
        "When the absolute difference between Normal and Abnormal probabilities is "
        "10 percentage points or less, the app asks organ-specific history questions. "
        "The resulting 0–100 context score is an experimental rule-based aid, not a "
        "validated medical risk score. Urgent warning symptoms override the score."
    )
    st.warning(
        "A NORMAL result does not rule out disease. An ABNORMAL result is not a "
        "diagnosis. Do not delay medical care because of an app result."
    )

    st.markdown("### Frequently asked questions")
    with st.expander("Why was my recording rejected?"):
        st.write(
            "The technical gate checks duration, signal level, silence and clipping. "
            "The current app expects at least 8 seconds for this quality check. "
            "Record again in a quieter environment with the microphone positioned steadily."
        )
    with st.expander("What if my microphone does not record?"):
        st.write(
            "Allow microphone access in your browser, check the operating-system input "
            "device, and reload the page. You can also use Upload audio file."
        )
    with st.expander("Are uploaded recordings saved permanently by this code?"):
        st.write(
            "This app code does not write uploaded recordings to a patient database. "
            "Audio is held during the app session and briefly written to a temporary "
            "file for decoding, which is removed after loading. Hosting infrastructure "
            "and browser behaviour are separate from this application code."
        )
    with st.expander("Why does the app only analyse 5 seconds?"):
        st.write(
            "The saved Keras models were trained with 5-second inputs. Changing the "
            "window length requires matching training and inference preprocessing and "
            "would require retraining the models."
        )

else:
    html_block(
        """
        <div class="hero-card">
            <div class="hero-badge">Project overview</div>
            <h1 class="hero-title">About StethoEdge</h1>
            <p class="hero-subtitle">An experimental AI-assisted heart and lung auscultation screening prototype.</p>
        </div>
        """
    )

    about_left, about_right = st.columns([1.15, 0.85], gap="large")
    with about_left:
        st.markdown("### What the project does")
        st.write(
            "StethoEdge accepts heart or lung audio, checks basic technical recording "
            "quality, converts the first five seconds into a log-Mel spectrogram, and "
            "runs the corresponding TensorFlow/Keras binary classifier. If the lung "
            "binary classifier returns ABNORMAL, a separate model estimates the "
            "abnormal sound subtype using the saved subtype-label file."
        )
        st.markdown("### Processing pipeline")
        pipeline = [
            "Live recording or audio-file upload",
            "Decode to mono audio at 16 kHz",
            "Technical duration / level / silence / clipping checks",
            "First 5 seconds → 64-band log-Mel spectrogram",
            "Heart or lung normal/abnormal model inference",
            "Lung subtype inference when the lung result is abnormal",
            "Organ-specific history form when probability gap ≤ 10 percentage points",
        ]
        for step in pipeline:
            st.markdown(f"- {step}")

        st.markdown("### Data and interpretation")
        st.write(
            "The project uses public heart-sound and respiratory-sound resources, "
            "including PhysioNet heart-sound material and ICBHI respiratory recordings, "
            "alongside the project's curated training folders. The current ICBHI lung "
            "binary import rule is based on crackle/wheeze annotations overlapping the "
            "first five seconds. That acoustic-event label is not the same as a patient "
            "diagnosis or a complete assessment of lung health."
        )

    with about_right:
        with st.container(border=True):
            st.markdown("### Model status")
            render_model_registry()
            if "lung_labels" in models:
                st.markdown("**Lung subtype labels**")
                for label_name in models["lung_labels"]:
                    st.write(f"• {format_label(str(label_name))}")

            st.markdown("---")
            st.markdown("### Technology")
            st.write("Python · Streamlit · TensorFlow/Keras · Librosa · NumPy · Matplotlib")

    st.markdown("### Limitations")
    limitation_cols = st.columns(3)
    limits = [
        ("Not a diagnosis", "The prototype is not clinically validated and cannot replace a clinician."),
        ("Dataset limits", "Dataset labels, recording devices, patient overlap and class coverage affect model performance."),
        ("History score", "The questionnaire score is an experimental rule-based context score, not a clinical risk prediction."),
    ]
    for col, (heading, body) in zip(limitation_cols, limits):
        with col:
            with st.container(border=True):
                st.markdown(f"**{heading}**")
                st.write(body)

    if TEAM_NAME or TEAM_MEMBERS or PROJECT_URL:
        st.markdown("### Team")
        if TEAM_NAME:
            st.write(f"**Team:** {TEAM_NAME}")
        if TEAM_MEMBERS:
            st.write("**Members:** " + ", ".join(TEAM_MEMBERS))
        if PROJECT_URL:
            st.link_button("Project link", PROJECT_URL)
