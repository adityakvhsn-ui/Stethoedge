"""
StethoEdge - offline Edge-AI cardiopulmonary SCREENING assistant (hackathon prototype)

Run:  streamlit run demo.py

IMPORTANT
- StethoEdge is a screening and triage aid, not a diagnostic replacement for a
  qualified healthcare professional.
- This hackathon prototype is NOT clinically validated.
- Without models/heart_model.tflite / models/lung_model.tflite the app runs a
  clearly labelled DEMO classifier whose output is simulated, not a prediction.

Code layout (all in this file so it can be pasted straight into demo.py):
  1. Config & paths
  2. i18n strings (English / Hindi / Kannada)
  3. Audio loading
  4. Signal Quality Index (prototype)
  5. Preprocessing & features (band-pass, noise gate, log-Mel)
  6. Model inference (TFLite engine + demo fallback + run_model)
  7. Risk mapping (probabilities -> Good / Medium / High Concern)
  8. Patient history -> next-step recommendation
  9. Local storage (SQLite store-and-forward queue)
 10. UI (Streamlit)
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="StethoEdge", page_icon="🩺", layout="wide")

# Fail with a readable message (not a traceback) if a dependency is missing.
try:
    import numpy as np
    import pandas as pd
    import librosa
    import soundfile as sf
    from scipy.signal import butter, sosfiltfilt, welch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as _err:  # pragma: no cover
    st.error(
        f"Missing dependency: {_err}. Install everything with:  "
        "`pip install -r requirements.txt`"
    )
    st.stop()


# =============================================================================
# 1. CONFIG & PATHS  (relative to this file - no hardcoded absolute paths)
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "screenings.db"

TARGET_SR = 4000      # all audio is resampled to this rate (Hz)
N_FFT = 512
HOP = 128
MAX_SECONDS = 30      # longer recordings are truncated

# Band-pass edges (Hz) used in preprocessing. Placeholders, tune with real data.
BANDS = {"heart": (25.0, 400.0), "lung": (100.0, 1800.0)}


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one screening model (the 'contract')."""
    kind: str
    filename: str
    window_s: float            # analysis window used for the demo / default
    n_mels: int                # default log-Mel bins (overridden by TFLite input shape)
    labels: tuple              # output label order expected from the model

    @property
    def path(self) -> Path:
        return MODEL_DIR / self.filename


MODEL_SPECS = {
    # heart output: [1,1] = P(abnormal)   or   [1,2] = (P(normal), P(abnormal))
    "heart": ModelSpec("heart", "heart_model.tflite", 5.0, 64, ("normal", "abnormal")),
    # lung output: [1,3] = independent sigmoids (normal, wheeze, crackle) -> multi-label,
    # so wheeze and crackle may co-occur.
    "lung": ModelSpec("lung", "lung_model.tflite", 8.0, 64, ("normal", "wheeze", "crackle")),
}

# --- Quality-gate thresholds: PROTOTYPE heuristics, NOT clinically validated ---
SQI_MIN_SECONDS = {"heart": 3.0, "lung": 5.0}
SQI_RECOMMENDED_SECONDS = {"heart": 8.0, "lung": 10.0}
SQI_MIN_RMS = 0.003
SQI_MAX_CLIP_FRACTION = 0.01
SQI_MAX_SILENCE_FRACTION = {"heart": 0.75, "lung": 0.60}
SQI_SIGNAL_BAND = {"heart": (20.0, 600.0), "lung": (80.0, 1200.0)}
SQI_MIN_BAND_EXCESS = 1.3
SQI_PASS_SCORE = 55.0

# --- Risk-mapping thresholds: UNCALIBRATED PLACEHOLDERS, not medical thresholds ---
RISK_THRESHOLDS = {"medium": 0.35, "high": 0.65}

POSITIONS = {
    "heart": ["aortic", "pulmonary", "tricuspid", "mitral"],
    "lung": ["upper_right", "upper_left", "lower_right", "lower_left"],
}
POSITION_HELP = {
    "aortic": "2nd intercostal space, right edge of the breastbone (sternum).",
    "pulmonary": "2nd intercostal space, left edge of the breastbone.",
    "tricuspid": "4th-5th intercostal space, left lower edge of the breastbone.",
    "mitral": "5th intercostal space, left side, in line with the middle of the collarbone (apex).",
    "upper_right": "Upper right chest, just below the collarbone.",
    "upper_left": "Upper left chest, just below the collarbone.",
    "lower_right": "Lower right side of the chest, about the level of the nipple line or below.",
    "lower_left": "Lower left side of the chest, about the level of the nipple line or below.",
}
# (x, y) positions on the schematic; patient's RIGHT is on the LEFT of the picture.
HEART_PTS = {"aortic": (84, 72), "pulmonary": (116, 72), "tricuspid": (108, 118), "mitral": (132, 140)}
LUNG_PTS = {"upper_right": (68, 72), "upper_left": (132, 72), "lower_right": (68, 142), "lower_left": (132, 142)}

AGE_GROUPS = ["<5", "5-17", "18-39", "40-59", "60+"]
CONDITIONS = ["heart", "asthma", "copd", "htn", "dm", "other", "none"]
SYMPTOMS = ["breathless", "cough", "wheezing", "fever", "fatigue", "palpitations",
            "chest_pain", "severe_breath", "fainting"]
RED_FLAG_SYMPTOMS = {"severe_breath", "fainting", "chest_pain"}


# =============================================================================
# 2. i18n  - add a language by appending one more column to each tuple.
#    Tuple order: (English, Hindi, Kannada). None -> falls back to English.
#    NOTE: Hindi/Kannada strings are first drafts - have a native speaker /
#    clinician review them before field use.
# =============================================================================
LANGS = {"English": "en", "हिन्दी": "hi", "ಕನ್ನಡ": "kn"}
LANG_INDEX = {"en": 0, "hi": 1, "kn": 2}

STRINGS = {
    "tagline": ("Offline cardiopulmonary screening assistant for Primary Health Centres",
                "प्राथमिक स्वास्थ्य केंद्रों के लिए ऑफ़लाइन हृदय-फेफड़ा स्क्रीनिंग सहायक",
                "ಪ್ರಾಥಮಿಕ ಆರೋಗ್ಯ ಕೇಂದ್ರಗಳಿಗಾಗಿ ಆಫ್‌ಲೈನ್ ಹೃದಯ-ಶ್ವಾಸಕೋಶ ತಪಾಸಣಾ ಸಹಾಯಕ"),
    "disclaimer": ("StethoEdge is a screening and triage aid, not a diagnostic replacement for a qualified healthcare professional.",
                   "StethoEdge एक स्क्रीनिंग और ट्राइएज सहायक है, योग्य स्वास्थ्य पेशेवर का निदान-विकल्प नहीं।",
                   "StethoEdge ಒಂದು ತಪಾಸಣೆ ಮತ್ತು ಟ್ರಯಾಜ್ ಸಹಾಯಕ; ಇದು ಅರ್ಹ ಆರೋಗ್ಯ ವೃತ್ತಿಪರರಿಗೆ ರೋಗನಿರ್ಣಯದ ಪರ್ಯಾಯವಲ್ಲ."),
    "proto_notice": ("Hackathon prototype - not clinically validated. No diagnosis, no guaranteed detection.",
                     "हैकथॉन प्रोटोटाइप - क्लिनिकल रूप से मान्य नहीं।",
                     "ಹ್ಯಾಕಥಾನ್ ಮಾದರಿ - ಕ್ಲಿನಿಕಲ್ ಆಗಿ ಮಾನ್ಯೀಕರಿಸಲಾಗಿಲ್ಲ."),
    "demo_title": ("DEMO MODE — TRAINED MODEL NOT CONNECTED",
                   "डेमो मोड — प्रशिक्षित मॉडल जुड़ा नहीं है",
                   "ಡೆಮೊ ಮೋಡ್ — ತರಬೇತಿ ಪಡೆದ ಮಾದರಿ ಸಂಪರ್ಕಿಸಿಲ್ಲ"),
    "demo_sub": ("These results are simulated for demonstration and are not medical predictions.",
                 "ये परिणाम प्रदर्शन के लिए सिम्युलेटेड हैं, चिकित्सीय पूर्वानुमान नहीं।",
                 "ಈ ಫಲಿತಾಂಶಗಳು ಪ್ರದರ್ಶನಕ್ಕಾಗಿ ಅನುಕರಿಸಿದವು, ವೈದ್ಯಕೀಯ ಮುನ್ಸೂಚನೆಗಳಲ್ಲ."),
    "step_word": ("STEP", "चरण", "ಹಂತ"),
    "step1": ("Screening Setup", "स्क्रीनिंग सेटअप", "ತಪಾಸಣೆ ಸೆಟಪ್"),
    "step2": ("Record", "रिकॉर्ड", "ರೆಕಾರ್ಡ್"),
    "step3": ("Check Recording Quality", "रिकॉर्डिंग गुणवत्ता जाँच", "ರೆಕಾರ್ಡಿಂಗ್ ಗುಣಮಟ್ಟ ಪರಿಶೀಲನೆ"),
    "step4": ("AI Screening", "AI स्क्रीनिंग", "AI ತಪಾಸಣೆ"),
    "step5": ("Result", "परिणाम", "ಫಲಿತಾಂಶ"),
    "step6": ("Additional History", "अतिरिक्त इतिहास", "ಹೆಚ್ಚುವರಿ ಇತಿಹಾಸ"),
    "step7": ("Next Action", "अगला कदम", "ಮುಂದಿನ ಕ್ರಮ"),
    "select_type": ("What do you want to screen?", "आप क्या जाँचना चाहते हैं?", "ನೀವು ಏನನ್ನು ತಪಾಸಣೆ ಮಾಡಲು ಬಯಸುತ್ತೀರಿ?"),
    "heart": ("Heart", "हृदय", "ಹೃದಯ"),
    "lung": ("Lungs", "फेफड़े", "ಶ್ವಾಸಕೋಶ"),
    "select_pos": ("Chest position", "छाती की स्थिति", "ಎದೆಯ ಸ್ಥಾನ"),
    "guide_title": ("Where to place the stethoscope", "स्टेथोस्कोप कहाँ रखें", "ಸ್ಟೆತೊಸ್ಕೋಪ್ ಎಲ್ಲಿ ಇಡಬೇಕು"),
    "diagram_note": ("Picture shows the patient from the front. The patient's right is on the left of the picture.",
                     None, None),
    "pos_aortic": ("Aortic", "महाधमनी (Aortic)", "ಮಹಾಧಮನಿ (Aortic)"),
    "pos_pulmonary": ("Pulmonary", "पल्मोनरी", "ಪಲ್ಮನರಿ"),
    "pos_tricuspid": ("Tricuspid", "ट्राइकस्पिड", "ಟ್ರೈಕಸ್ಪಿಡ್"),
    "pos_mitral": ("Mitral", "माइट्रल", "ಮಿಟ್ರಲ್"),
    "pos_upper_right": ("Upper right", "ऊपर दायाँ", "ಮೇಲಿನ ಬಲ"),
    "pos_upper_left": ("Upper left", "ऊपर बायाँ", "ಮೇಲಿನ ಎಡ"),
    "pos_lower_right": ("Lower right", "नीचे दायाँ", "ಕೆಳಗಿನ ಬಲ"),
    "pos_lower_left": ("Lower left", "नीचे बायाँ", "ಕೆಳಗಿನ ಎಡ"),
    "start_btn": ("Start recording →", "रिकॉर्डिंग शुरू करें →", "ರೆಕಾರ್ಡಿಂಗ್ ಪ್ರಾರಂಭಿಸಿ →"),
    "change_setup": ("← Change setup", "← सेटअप बदलें", "← ಸೆಟಪ್ ಬದಲಿಸಿ"),
    "record_help": ("Record 8-10 seconds in a quiet room. Hold the stethoscope still and ask the patient to stay quiet. For lungs, ask for slow deep breaths through the mouth.",
                    None, None),
    "tab_record": ("🎙️ Record", "🎙️ रिकॉर्ड", "🎙️ ರೆಕಾರ್ಡ್"),
    "tab_upload": ("📁 Upload file", "📁 फ़ाइल अपलोड", "📁 ಫೈಲ್ ಅಪ್‌ಲೋಡ್"),
    "record_label": ("Press to record the chest sound", "छाती की आवाज़ रिकॉर्ड करने के लिए दबाएँ", "ಎದೆಯ ಶಬ್ದ ರೆಕಾರ್ಡ್ ಮಾಡಲು ಒತ್ತಿ"),
    "mic_help": ("If the microphone does not start: allow microphone access in the browser (needs localhost or HTTPS), check that a microphone is connected, or use the Upload tab.",
                 None, None),
    "no_audio_input": ("This Streamlit version has no browser recorder. Upgrade Streamlit (>=1.40) or use the Upload tab.",
                       None, None),
    "upload_label": ("Upload a recording (WAV recommended)", "रिकॉर्डिंग अपलोड करें (WAV बेहतर)", "ರೆಕಾರ್ಡಿಂಗ್ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ (WAV ಉತ್ತಮ)"),
    "listen": ("Listen to the recording", "रिकॉर्डिंग सुनें", "ರೆಕಾರ್ಡಿಂಗ್ ಆಲಿಸಿ"),
    "read_error": ("This audio could not be read. The file may be corrupted or in an unsupported format. Please record again or upload a WAV file.",
                   "यह ऑडियो पढ़ा नहीं जा सका। कृपया दोबारा रिकॉर्ड करें या WAV फ़ाइल अपलोड करें।",
                   "ಈ ಆಡಿಯೋ ಓದಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ರೆಕಾರ್ಡ್ ಮಾಡಿ ಅಥವಾ WAV ಫೈಲ್ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ."),
    "sqi_proto": ("Prototype quality check - not a clinically validated SQI.",
                  "प्रोटोटाइप गुणवत्ता जाँच - क्लिनिकल रूप से मान्य SQI नहीं।",
                  "ಮಾದರಿ ಗುಣಮಟ್ಟ ಪರಿಶೀಲನೆ - ಕ್ಲಿನಿಕಲ್ ಆಗಿ ಮಾನ್ಯೀಕರಿಸಿದ SQI ಅಲ್ಲ."),
    "poor_recording": ("POOR RECORDING", "खराब रिकॉर्डिंग", "ಕಳಪೆ ರೆಕಾರ್ಡಿಂಗ್"),
    "reposition": ("Please reposition the stethoscope and record again.",
                   "कृपया स्टेथोस्कोप की स्थिति बदलें और दोबारा रिकॉर्ड करें।",
                   "ದಯವಿಟ್ಟು ಸ್ಟೆತೊಸ್ಕೋಪ್ ಸ್ಥಾನ ಬದಲಿಸಿ ಮತ್ತು ಮತ್ತೊಮ್ಮೆ ರೆಕಾರ್ಡ್ ಮಾಡಿ."),
    "record_again": ("🔄 Record again", "🔄 दोबारा रिकॉर्ड करें", "🔄 ಮತ್ತೊಮ್ಮೆ ರೆಕಾರ್ಡ್ ಮಾಡಿ"),
    "quality_ok": ("RECORDING QUALITY ACCEPTABLE", "रिकॉर्डिंग की गुणवत्ता स्वीकार्य", "ರೆಕಾರ್ಡಿಂಗ್ ಗುಣಮಟ್ಟ ಸ್ವೀಕಾರಾರ್ಹ"),
    "quality_score": ("Quality score", "गुणवत्ता स्कोर", "ಗುಣಮಟ್ಟ ಸ್ಕೋರ್"),
    "quality_details": ("Quality check details", None, None),
    "issue_too_short": ("The recording is too short.", "रिकॉर्डिंग बहुत छोटी है।", "ರೆಕಾರ್ಡಿಂಗ್ ತುಂಬಾ ಚಿಕ್ಕದಾಗಿದೆ."),
    "issue_too_weak": ("The signal is too weak - press the stethoscope firmly on bare skin or thin clothing.",
                       "सिग्नल बहुत कमज़ोर है - स्टेथोस्कोप को मज़बूती से दबाएँ।", None),
    "issue_clipping": ("The signal is distorted (clipping) - reduce pressure or microphone gain.",
                       "सिग्नल विकृत है (क्लिपिंग)।", None),
    "issue_silence": ("There is too much silence - the stethoscope may have lost contact.",
                      "रिकॉर्डिंग में बहुत अधिक चुप्पी है।", None),
    "issue_noisy": ("The recording is too noisy or does not look like a chest sound - move to a quieter place.",
                    "रिकॉर्डिंग बहुत शोर वाली है।", None),
    "run_screening": ("Run AI screening", "AI स्क्रीनिंग चलाएँ", "AI ತಪಾಸಣೆ ನಡೆಸಿ"),
    "screening_busy": ("Analysing the recording on this device...", "इस डिवाइस पर रिकॉर्डिंग का विश्लेषण...", None),
    "good": ("GOOD", "अच्छा", "ಉತ್ತಮ"),
    "medium": ("MEDIUM", "मध्यम", "ಮಧ್ಯಮ"),
    "high": ("HIGH CONCERN", "उच्च चिंता", "ಹೆಚ್ಚಿನ ಕಳವಳ"),
    "good_meaning": ("No immediate concern detected by the screening system.",
                     "स्क्रीनिंग प्रणाली ने कोई तत्काल चिंता नहीं पाई।",
                     "ತಪಾಸಣಾ ವ್ಯವಸ್ಥೆ ಯಾವುದೇ ತಕ್ಷಣದ ಕಳವಳವನ್ನು ಪತ್ತೆ ಮಾಡಿಲ್ಲ."),
    "medium_meaning": ("Additional patient context is required before deciding the next step.",
                       "अगला कदम तय करने से पहले रोगी की अतिरिक्त जानकारी आवश्यक है।",
                       "ಮುಂದಿನ ಕ್ರಮ ನಿರ್ಧರಿಸುವ ಮೊದಲು ರೋಗಿಯ ಹೆಚ್ಚುವರಿ ಮಾಹಿತಿ ಅಗತ್ಯ."),
    "high_meaning": ("The screening result requires clinical attention.",
                     "स्क्रीनिंग परिणाम पर क्लिनिकल ध्यान आवश्यक है।",
                     "ತಪಾಸಣೆ ಫಲಿತಾಂಶಕ್ಕೆ ಕ್ಲಿನಿಕಲ್ ಗಮನ ಅಗತ್ಯ."),
    "good_next": ("Continue according to the local clinical workflow.",
                  "स्थानीय क्लिनिकल कार्यप्रवाह के अनुसार आगे बढ़ें।",
                  "ಸ್ಥಳೀಯ ಕ್ಲಿನಿಕಲ್ ಕಾರ್ಯವಿಧಾನದ ಪ್ರಕಾರ ಮುಂದುವರಿಸಿ."),
    "high_next": ("Clinical assessment / referral recommended according to local protocol.",
                  "स्थानीय प्रोटोकॉल के अनुसार क्लिनिकल जाँच / रेफ़रल की सलाह दी जाती है।",
                  "ಸ್ಥಳೀಯ ಪ್ರೋಟೋಕಾಲ್ ಪ್ರಕಾರ ಕ್ಲಿನಿಕಲ್ ಮೌಲ್ಯಮಾಪನ / ರೆಫರಲ್ ಶಿಫಾರಸು."),
    "simulated_tag": ("SIMULATED RESULT — NOT A MEDICAL PREDICTION", "सिम्युलेटेड परिणाम — चिकित्सीय पूर्वानुमान नहीं", None),
    "viz_title": ("What was analysed", "क्या विश्लेषण किया गया", None),
    "viz_note": ("Visualization only. The highlighted window is chosen by signal energy, not by the AI model. This is not a model-derived explanation.",
                 None, None),
    "explain_na": ("Model-derived explanation (e.g. which segment drove the prediction): not available in this prototype yet.",
                   None, None),
    "history_intro": ("Please enter the patient's history and current symptoms so we can suggest the next step.",
                      "अगला कदम सुझाने के लिए रोगी का इतिहास और वर्तमान लक्षण दर्ज करें।",
                      "ಮುಂದಿನ ಕ್ರಮ ಸೂಚಿಸಲು ರೋಗಿಯ ಇತಿಹಾಸ ಮತ್ತು ಪ್ರಸ್ತುತ ಲಕ್ಷಣಗಳನ್ನು ನಮೂದಿಸಿ."),
    "age_group": ("Age group (years)", "आयु वर्ग (वर्ष)", "ವಯೋಮಾನ (ವರ್ಷ)"),
    "known_cond": ("Known conditions", "ज्ञात बीमारियाँ", "ತಿಳಿದಿರುವ ಕಾಯಿಲೆಗಳು"),
    "cond_heart": ("Heart disease", "हृदय रोग", "ಹೃದ್ರೋಗ"),
    "cond_asthma": ("Asthma", "अस्थमा", "ಅಸ್ತಮಾ"),
    "cond_copd": ("COPD", "COPD", "COPD"),
    "cond_htn": ("Hypertension", "उच्च रक्तचाप", "ಅಧಿಕ ರಕ್ತದೊತ್ತಡ"),
    "cond_dm": ("Diabetes", "मधुमेह", "ಮಧುಮೇಹ"),
    "cond_other": ("Other relevant condition", "अन्य संबंधित बीमारी", "ಇತರೆ ಸಂಬಂಧಿತ ಕಾಯಿಲೆ"),
    "cond_none": ("None known", "कोई ज्ञात नहीं", "ಯಾವುದೂ ತಿಳಿದಿಲ್ಲ"),
    "tobacco": ("Tobacco use (smoking / chewing)", "तंबाकू का उपयोग", "ತಂಬಾಕು ಬಳಕೆ"),
    "yes": ("Yes", "हाँ", "ಹೌದು"),
    "no": ("No", "नहीं", "ಇಲ್ಲ"),
    "symptoms": ("Current symptoms", "वर्तमान लक्षण", "ಪ್ರಸ್ತುತ ಲಕ್ಷಣಗಳು"),
    "sym_breathless": ("Breathlessness", "साँस फूलना", "ಉಸಿರಾಟದ ತೊಂದರೆ"),
    "sym_cough": ("Persistent cough", "लगातार खाँसी", "ನಿರಂತರ ಕೆಮ್ಮು"),
    "sym_wheezing": ("Wheezing", "घरघराहट", "ಉಬ್ಬಸ"),
    "sym_fever": ("Fever", "बुखार", "ಜ್ವರ"),
    "sym_fatigue": ("Fatigue", "थकान", "ಆಯಾಸ"),
    "sym_palpitations": ("Palpitations", "धड़कन तेज़ होना", "ಹೃದಯ ಬಡಿತ ಹೆಚ್ಚಳ"),
    "sym_chest_pain": ("Chest discomfort", "सीने में असुविधा", "ಎದೆ ಅಸ್ವಸ್ಥತೆ"),
    "sym_severe_breath": ("Severe breathing difficulty", "साँस लेने में गंभीर कठिनाई", "ತೀವ್ರ ಉಸಿರಾಟದ ತೊಂದರೆ"),
    "sym_fainting": ("Fainting", "बेहोशी", "ಮೂರ್ಛೆ"),
    "submit_history": ("Get next-step recommendation", "अगले कदम की सलाह लें", "ಮುಂದಿನ ಕ್ರಮದ ಶಿಫಾರಸು ಪಡೆಯಿರಿ"),
    "rec_prompt": ("Prompt clinical assessment recommended", "तुरंत क्लिनिकल जाँच की सलाह दी जाती है", "ತಕ್ಷಣದ ಕ್ಲಿನಿಕಲ್ ಮೌಲ್ಯಮಾಪನ ಶಿಫಾರಸು"),
    "rec_doctor": ("Doctor consultation recommended", "डॉक्टर से परामर्श की सलाह", "ವೈದ್ಯರ ಸಲಹೆ ಶಿಫಾರಸು"),
    "rec_protocol": ("Follow local clinical protocol", "स्थानीय क्लिनिकल प्रोटोकॉल का पालन करें", "ಸ್ಥಳೀಯ ಕ್ಲಿನಿಕಲ್ ಪ್ರೋಟೋಕಾಲ್ ಅನುಸರಿಸಿ"),
    "rec_none": ("No additional high-risk information entered", "कोई अतिरिक्त उच्च-जोखिम जानकारी दर्ज नहीं", "ಹೆಚ್ಚುವರಿ ಹೆಚ್ಚಿನ ಅಪಾಯದ ಮಾಹಿತಿ ನಮೂದಿಸಿಲ್ಲ"),
    "rec_note": ("This suggestion is a simple rule-based prototype and is not a validated clinical decision.",
                 None, None),
    "final_note": ("The final decision belongs to a qualified healthcare professional.",
                   "अंतिम निर्णय योग्य स्वास्थ्य पेशेवर का है।",
                   "ಅಂತಿಮ ನಿರ್ಧಾರ ಅರ್ಹ ಆರೋಗ್ಯ ವೃತ್ತಿಪರರದ್ದು."),
    "new_screening": ("Start new screening / next position", "नई स्क्रीनिंग / अगली स्थिति", "ಹೊಸ ತಪಾಸಣೆ / ಮುಂದಿನ ಸ್ಥಾನ"),
    "saved_note": ("Saved on this device only (no audio, no name). Pending upload to a clinician queue.",
                   None, None),
}


def t(key: str) -> str:
    """Translate a UI string using the selected language (English fallback)."""
    entry = STRINGS.get(key)
    if entry is None:
        return key
    idx = LANG_INDEX.get(st.session_state.get("lang_code", "en"), 0)
    value = entry[idx] if idx < len(entry) else None
    return value or entry[0]


# =============================================================================
# 3. AUDIO LOADING
# =============================================================================
def load_audio(data: bytes, filename: str = "") -> tuple:
    """Decode bytes -> (mono float32 array at TARGET_SR, None) or (None, error code).

    Never raises: bad / corrupted / unsupported audio just returns an error.
    """
    if not data or len(data) < 64:
        return None, "empty"
    y, sr = None, None
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:
        # Fallback for formats libsndfile cannot read (e.g. some mp3/m4a builds).
        path = None
        try:
            suffix = Path(filename).suffix or ".wav"
            fd, path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            with open(path, "wb") as fh:
                fh.write(data)
            y, sr = librosa.load(path, sr=None, mono=True)
        except Exception:
            return None, "unreadable"
        finally:
            if path and os.path.exists(path):
                try:
                    os.remove(path)  # do not leave patient audio in temp files
                except OSError:
                    pass
    try:
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if y.size == 0 or not sr or sr <= 0:
            return None, "unreadable"
        if not np.all(np.isfinite(y)):
            return None, "corrupted"
        if int(sr) != TARGET_SR:
            y = librosa.resample(y, orig_sr=int(sr), target_sr=TARGET_SR)
        y = y[: MAX_SECONDS * TARGET_SR]
        y = y - float(np.mean(y))  # remove DC offset
        return y.astype(np.float32), None
    except Exception:
        return None, "unreadable"


# =============================================================================
# 4. SIGNAL QUALITY INDEX (PROTOTYPE - not clinically validated)
# =============================================================================
@dataclass
class SQIResult:
    score: float                      # 0-100 heuristic score
    passed: bool
    issues: list = field(default_factory=list)   # issue codes
    metrics: dict = field(default_factory=dict)


def compute_sqi(y, sr: int, kind: str) -> SQIResult:
    """Heuristic quality gate: duration, level, clipping, silence, band energy."""
    issues = []
    duration = len(y) / sr
    min_dur = SQI_MIN_SECONDS[kind]
    rec_dur = SQI_RECOMMENDED_SECONDS[kind]
    if duration < min_dur:
        issues.append("too_short")
    if duration < 0.3:  # too little data to compute anything else
        return SQIResult(0.0, False, issues, {"duration_s": round(duration, 2)})

    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    if rms < SQI_MIN_RMS:
        issues.append("too_weak")

    # Clipping: many samples stuck at the peak when the peak is near full scale.
    peak = float(np.max(np.abs(y)))
    clip_frac = float(np.mean(np.abs(y) >= 0.99 * peak)) if peak >= 0.95 else 0.0
    if clip_frac > SQI_MAX_CLIP_FRACTION:
        issues.append("clipping")

    # Silence: fraction of 50 ms frames far below the loud-frame level.
    frame = int(0.05 * sr)
    n_frames = len(y) // frame
    frame_rms = np.sqrt(np.mean(y[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1))
    thresh = max(1e-4, 0.10 * float(np.percentile(frame_rms, 95)))
    silence_frac = float(np.mean(frame_rms < thresh))
    max_sil = SQI_MAX_SILENCE_FRACTION[kind]
    if silence_frac > max_sil:
        issues.append("silence")

    # Spectral shape: chest sounds concentrate energy in a low band. A nearly flat
    # spectrum (energy ~ proportional to bandwidth) suggests noise / no contact.
    f, p = welch(y, fs=sr, nperseg=min(512, len(y)))
    total_mask = (f >= 20) & (f <= sr / 2 - 1)
    lo, hi = SQI_SIGNAL_BAND[kind]
    band_mask = (f >= lo) & (f <= hi)
    total_e = float(np.sum(p[total_mask])) + 1e-12
    band_ratio = float(np.sum(p[band_mask])) / total_e
    expected = (hi - lo) / ((sr / 2 - 1) - 20)       # ratio a flat spectrum would give
    band_excess = band_ratio / expected
    if band_excess < SQI_MIN_BAND_EXCESS and "too_weak" not in issues:
        issues.append("noisy")

    # Component scores 0..1 -> weighted 0..100 (weights are arbitrary prototype choices).
    s_dur = min(duration / rec_dur, 1.0)
    s_lvl = float(np.clip((rms - SQI_MIN_RMS) / 0.01, 0, 1))
    s_clip = 1.0 - min(clip_frac / SQI_MAX_CLIP_FRACTION, 1.0)
    s_sil = float(np.clip(1.0 - silence_frac / max_sil, 0, 1))
    s_band = float(np.clip((band_excess - 1.0) / 1.5, 0, 1))
    score = 100.0 * (0.20 * s_dur + 0.25 * s_lvl + 0.20 * s_clip + 0.15 * s_sil + 0.20 * s_band)

    passed = (not issues) and score >= SQI_PASS_SCORE
    if not passed and not issues:
        issues.append("noisy")  # low overall score without a specific cause
    metrics = {
        "duration_s": round(duration, 2), "rms": round(rms, 4),
        "clipping_fraction": round(clip_frac, 4), "silence_fraction": round(silence_frac, 3),
        "band_energy_excess": round(band_excess, 2),
    }
    return SQIResult(round(score, 1), passed, issues, metrics)


# =============================================================================
# 5. PREPROCESSING & FEATURE EXTRACTION
# =============================================================================
def bandpass(y, sr: int, lo: float, hi: float, order: int = 4):
    hi = min(hi, 0.95 * sr / 2)
    sos = butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, y)


def reduce_noise(y, strength: float = 1.5):
    """Very basic spectral gate: subtract a per-bin noise-floor estimate.
    Placeholder for a better denoiser; robustness is NOT validated."""
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    mag, phase = np.abs(stft), np.angle(stft)
    noise_floor = np.percentile(mag, 10, axis=1, keepdims=True)
    cleaned = np.maximum(mag - strength * noise_floor, 0.1 * mag)
    return librosa.istft(cleaned * np.exp(1j * phase), hop_length=HOP, length=len(y))


def preprocess(y, sr: int, kind: str):
    """band-pass -> noise gate -> peak normalise."""
    lo, hi = BANDS[kind]
    y = bandpass(y, sr, lo, hi)
    y = reduce_noise(y)
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)


def select_window(y, win_len: int, sr: int) -> tuple:
    """Pick the highest-energy window of length win_len (pads with zeros if short).
    This is a simple placeholder for future model-driven segment selection."""
    if len(y) <= win_len:
        return 0, np.pad(y, (0, win_len - len(y)))
    csum = np.concatenate([[0.0], np.cumsum(y.astype(np.float64) ** 2)])
    starts = np.arange(0, len(y) - win_len + 1, max(1, int(0.25 * sr)))
    energy = csum[starts + win_len] - csum[starts]
    s = int(starts[int(np.argmax(energy))])
    return s, y[s: s + win_len]


def log_mel(y, sr: int, n_mels: int, n_frames: int):
    """Log-Mel spectrogram in dB, shape (n_mels, n_frames)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=n_mels, fmin=20, fmax=sr / 2)
    db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    if db.shape[1] < n_frames:
        db = np.pad(db, ((0, 0), (0, n_frames - db.shape[1])), constant_values=db.min())
    return db[:, :n_frames].astype(np.float32)


# =============================================================================
# 6. MODEL INFERENCE  (TFLite engine + demo fallback)
# =============================================================================
class ModelContractError(Exception):
    """Raised when a .tflite file does not match the documented input/output contract."""


def _import_interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter  # light-weight runtime
        return Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter  # full TF
            return Interpreter
        except ImportError:
            return None


class TFLiteEngine:
    """Thin wrapper around a TFLite interpreter.

    INPUT CONTRACT : float32 [1, n_mels, n_frames, 1] - per-sample standardised log-Mel (dB)
                     (int8/uint8 quantised inputs are handled automatically)
    OUTPUT CONTRACT: probabilities in [0, 1] - see parse_output().
    """

    def __init__(self, path: str):
        interpreter_cls = _import_interpreter()
        if interpreter_cls is None:
            raise RuntimeError("No TFLite runtime installed (pip install tflite-runtime, or tensorflow).")
        self.interp = interpreter_cls(model_path=path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        shape = [int(s) for s in self.inp["shape"]]
        if len(shape) != 4 or shape[3] != 1 or shape[1] <= 0 or shape[2] <= 0:
            raise ModelContractError(f"Expected input [1, n_mels, n_frames, 1], got {shape}")
        self.n_mels, self.n_frames = shape[1], shape[2]

    def predict(self, mel_std):
        x = mel_std.astype(np.float32)[None, :, :, None]
        dtype = self.inp["dtype"]
        if dtype != np.float32:  # quantised model
            scale, zero = self.inp["quantization"]
            if scale:
                info = np.iinfo(dtype)
                x = np.clip(np.round(x / scale + zero), info.min, info.max).astype(dtype)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        raw = self.interp.get_tensor(self.out["index"])
        if raw.dtype != np.float32:
            scale, zero = self.out["quantization"]
            raw = (raw.astype(np.float32) - zero) * (scale or 1.0)
        return raw.astype(np.float32).reshape(-1)


@st.cache_resource(show_spinner=False)
def get_engine(path: str, mtime: float) -> TFLiteEngine:
    """Cached per (path, modification time) so replacing the file reloads the model."""
    return TFLiteEngine(path)


def parse_output(kind: str, vec) -> dict:
    """Model output vector -> {label: probability}. Enforces the output contract."""
    if np.any(vec < -1e-3) or np.any(vec > 1 + 1e-3):
        raise ModelContractError("Model outputs must be probabilities in [0,1] (apply sigmoid/softmax inside the model).")
    vec = np.clip(vec, 0.0, 1.0)
    if kind == "heart":
        if vec.size == 1:
            p_abn = float(vec[0])
        elif vec.size == 2:
            p_abn = float(vec[1])
        else:
            raise ModelContractError(f"Heart model must output 1 or 2 values, got {vec.size}")
        return {"normal": 1.0 - p_abn, "abnormal": p_abn}
    if vec.size != 3:
        raise ModelContractError(f"Lung model must output 3 values (normal, wheeze, crackle), got {vec.size}")
    return {"normal": float(vec[0]), "wheeze": float(vec[1]), "crackle": float(vec[2])}


def demo_classify(kind: str, y, scenario: str) -> dict:
    """DEMO CLASSIFIER - NOT A MEDICAL PREDICTION.

    Produces a simulated score so all three result states can be demonstrated.
    'auto' derives a repeatable pseudo-random score from the audio bytes, which
    has NO relationship to the patient's health.
    """
    fixed = {"good": 0.12, "medium": 0.50, "high": 0.85}
    if scenario in fixed:
        s = fixed[scenario]
    else:
        digest = hashlib.sha1(np.round(y[:4000] * 1000).astype(np.int16).tobytes()).hexdigest()
        s = (int(digest[:8], 16) % 1000) / 1000.0
    if kind == "heart":
        return {"normal": 1 - s, "abnormal": s}
    return {"normal": 1 - s, "wheeze": s, "crackle": 0.0}


@dataclass
class ModelResult:
    mode: str                 # "tflite" or "demo"
    kind: str
    probs: dict
    y_proc: object            # preprocessed full recording (for display)
    sr: int
    win_start_s: float
    win_s: float
    mel_db: object            # log-Mel of the analysed window
    notes: list = field(default_factory=list)


def model_file_present(kind: str) -> bool:
    return MODEL_SPECS[kind].path.exists()


def run_model(kind: str, y, sr: int, demo_scenario: str = "auto") -> ModelResult:
    """Single entry point for screening inference.

    Pipeline: preprocess -> select window -> log-Mel -> TFLite model (or DEMO) -> probabilities.
    Replace nothing here when real models arrive: just drop the .tflite files in models/.
    """
    spec = MODEL_SPECS[kind]
    notes = []
    y_proc = preprocess(y, sr, kind)

    engine = None
    if spec.path.exists():
        try:
            engine = get_engine(str(spec.path), spec.path.stat().st_mtime)
        except Exception as exc:  # bad/incompatible model -> clear message + demo fallback
            notes.append(f"Could not use {spec.filename}: {exc}. Falling back to DEMO MODE.")

    if engine is not None:
        n_mels, n_frames = engine.n_mels, engine.n_frames
    else:
        n_mels = spec.n_mels
        n_frames = 1 + int(spec.window_s * sr) // HOP
    win_len = (n_frames - 1) * HOP
    start, segment = select_window(y_proc, win_len, sr)
    mel_db = log_mel(segment, sr, n_mels, n_frames)

    if engine is not None:
        try:
            mel_std = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            probs = parse_output(kind, engine.predict(mel_std))
            mode = "tflite"
        except Exception as exc:
            notes.append(f"Model inference failed: {exc}. Falling back to DEMO MODE.")
            probs, mode = demo_classify(kind, y_proc, demo_scenario), "demo"
    else:
        probs, mode = demo_classify(kind, y_proc, demo_scenario), "demo"

    return ModelResult(mode, kind, probs, y_proc, sr, start / sr, win_len / sr, mel_db, notes)


# =============================================================================
# 7. RISK MAPPING  (isolated so it can be calibrated later with clinician-labelled data)
# =============================================================================
def abnormality_score(kind: str, probs: dict) -> float:
    """Collapse model probabilities into one 0-1 score. Lung labels are NOT mutually
    exclusive, so the score is the max over the abnormal labels."""
    if kind == "heart":
        return float(probs["abnormal"])
    return float(max(probs["wheeze"], probs["crackle"]))


def map_probabilities_to_risk(kind: str, probs: dict) -> tuple:
    """Return (level, score) with level in {"good", "medium", "high"}.

    RISK_THRESHOLDS are uncalibrated placeholders - NOT medically validated.
    Calibrate (sensitivity-first) on validation + clinician-labelled data before any real use.
    """
    score = abnormality_score(kind, probs)
    if score >= RISK_THRESHOLDS["high"]:
        return "high", score
    if score >= RISK_THRESHOLDS["medium"]:
        return "medium", score
    return "good", score


# =============================================================================
# 8. PATIENT HISTORY -> NEXT-STEP RECOMMENDATION (simple prototype rules)
# =============================================================================
def recommend_next_step(conditions: set, tobacco: bool, symptoms: set) -> str:
    """Returns one of: prompt / doctor / protocol / none.
    Rule-based placeholder, NOT a validated clinical decision rule."""
    conditions = conditions - {"none"}
    if symptoms & RED_FLAG_SYMPTOMS:
        return "prompt"
    if conditions & {"heart", "asthma", "copd"} or len(symptoms) >= 2:
        return "doctor"
    if symptoms or conditions or tobacco:
        return "protocol"
    return "none"


# =============================================================================
# 9. LOCAL STORAGE - SQLite store-and-forward queue (no names, no audio)
# =============================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS screenings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc TEXT NOT NULL,
    screening_type TEXT NOT NULL,
    chest_position TEXT NOT NULL,
    sqi_score REAL,
    sqi_passed INTEGER NOT NULL,
    result TEXT NOT NULL,          -- GOOD | MEDIUM | HIGH_CONCERN | POOR_RECORDING
    model_mode TEXT,               -- tflite | demo  (demo rows are NOT real predictions)
    age_group TEXT,
    next_action TEXT,
    sync_status TEXT NOT NULL      -- pending | synced | local_only
)
"""


def _db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return closing(sqlite3.connect(DB_PATH))


def init_db() -> None:
    try:
        with _db() as con:
            con.execute(SCHEMA)
            con.commit()
    except sqlite3.Error as exc:
        st.warning(f"Local storage unavailable: {exc}")


def insert_screening(kind, position, sqi: SQIResult, result: str, model_mode, next_action=None):
    sync = "local_only" if result == "POOR_RECORDING" else "pending"
    try:
        with _db() as con:
            cur = con.execute(
                "INSERT INTO screenings (created_utc, screening_type, chest_position, sqi_score, "
                "sqi_passed, result, model_mode, next_action, sync_status) VALUES (?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), kind, position,
                 sqi.score, int(sqi.passed), result, model_mode, next_action, sync))
            con.commit()
            return cur.lastrowid
    except sqlite3.Error:
        return None


def update_history(row_id, age_group: str, next_action: str) -> None:
    if row_id is None:
        return
    try:
        with _db() as con:
            con.execute("UPDATE screenings SET age_group=?, next_action=? WHERE id=?",
                        (age_group, next_action, row_id))
            con.commit()
    except sqlite3.Error:
        pass


def read_screenings(limit: int | None = None):
    try:
        with _db() as con:
            q = "SELECT * FROM screenings ORDER BY id DESC"
            if limit:
                q += f" LIMIT {int(limit)}"
            return pd.read_sql_query(q, con)
    except Exception:
        return pd.DataFrame()


def pending_count() -> int:
    try:
        with _db() as con:
            return con.execute("SELECT COUNT(*) FROM screenings WHERE sync_status='pending'").fetchone()[0]
    except sqlite3.Error:
        return 0


def simulate_sync() -> int:
    """DEMO of the 'later synchronisation' step: only flips a flag locally. Nothing is sent anywhere."""
    try:
        with _db() as con:
            n = con.execute("UPDATE screenings SET sync_status='synced' WHERE sync_status='pending'").rowcount
            con.commit()
            return n
    except sqlite3.Error:
        return 0


def delete_all_data() -> None:
    try:
        with _db() as con:
            con.execute("DELETE FROM screenings")
            con.commit()
            con.execute("VACUUM")
    except sqlite3.Error:
        pass


# =============================================================================
# 10. UI
# =============================================================================
CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1100px;}
.sx-title {font-size: 2rem; font-weight: 800; letter-spacing: -0.5px; margin: 0;}
.sx-sub {color: #64748b; margin: 0 0 .6rem 0;}
.sx-stepper {display: flex; gap: 6px; flex-wrap: wrap; margin: .4rem 0 1.2rem 0;}
.sx-pill {display: flex; align-items: center; gap: 6px; padding: 5px 11px; border-radius: 999px;
          font-size: .78rem; border: 1px solid #cbd5e1; color: #64748b; background: transparent;}
.sx-pill .n {display: inline-flex; width: 20px; height: 20px; border-radius: 50%; align-items: center;
             justify-content: center; font-weight: 700; font-size: .72rem; background: #e2e8f0; color: #334155;}
.sx-pill.done {border-color: #86efac; color: #166534;}
.sx-pill.done .n {background: #22c55e; color: white;}
.sx-pill.now {border-color: #2563eb; background: #2563eb; color: white; font-weight: 700;}
.sx-pill.now .n {background: white; color: #2563eb;}
.sx-step {font-size: 1.05rem; font-weight: 700; margin: 1.4rem 0 .4rem 0; padding-bottom: 4px;
          border-bottom: 2px solid #e2e8f0;}
.sx-card {border-radius: 14px; padding: 1.1rem 1.3rem; margin: .6rem 0; border: 2px solid; }
.sx-card .big {font-size: 2.6rem; font-weight: 900; line-height: 1.1; letter-spacing: -1px;}
.sx-card .mid {font-size: 1.15rem; font-weight: 600; margin-top: .3rem;}
.sx-card .small {font-size: .9rem; margin-top: .4rem; opacity: .9;}
.sx-good {background: #e8f7ee; border-color: #1e9e5a; color: #0b4f2b;}
.sx-medium {background: #fff6d6; border-color: #e0a100; color: #6b4a00;}
.sx-high {background: #fde8e8; border-color: #d62828; color: #7a1010;}
.sx-info {background: #eef4ff; border-color: #2563eb; color: #1e3a8a;}
.sx-demo {background: #fff1f2; border: 2px dashed #e11d48; color: #9f1239; border-radius: 12px;
          padding: .7rem 1rem; margin-bottom: .8rem;}
.sx-demo b {font-size: 1.05rem;}
.sx-disc {font-size: .82rem; color: #64748b; border-left: 3px solid #94a3b8; padding-left: .7rem; margin: .5rem 0 1rem 0;}
.sx-badge {display: inline-block; font-size: .72rem; font-weight: 800; padding: 2px 8px; border-radius: 6px;
           background: #e11d48; color: white; margin-top: .5rem;}
</style>
"""


def init_state() -> None:
    defaults = dict(
        lang_code="en", setup_done=False, rec_n=0, screen_n=0,
        analysis=None, screening=None, row_id=None,
        history_done=False, history_rec=None, demo_scenario="auto",
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def current_kind() -> str:
    return st.session_state.get("kind_w", "heart")


def current_position() -> str:
    kind = current_kind()
    return st.session_state.get(f"pos_{kind}", POSITIONS[kind][0])


def reset_recording() -> None:
    ss = st.session_state
    ss.rec_n += 1
    ss.analysis = None
    ss.screening = None
    ss.row_id = None
    ss.history_done = False
    ss.history_rec = None


def reset_screening() -> None:
    reset_recording()
    st.session_state.screen_n += 1
    st.session_state.setup_done = False


def step_heading(n: int) -> None:
    st.markdown(f'<div class="sx-step">{t("step_word")} {n} — {t(f"step{n}")}</div>', unsafe_allow_html=True)


def card(css: str, big: str, mid: str = "", small: str = "", badge: str = "") -> None:
    html = f'<div class="sx-card sx-{css}"><div class="big">{big}</div>'
    if mid:
        html += f'<div class="mid">{mid}</div>'
    if small:
        html += f'<div class="small">{small}</div>'
    if badge:
        html += f'<span class="sx-badge">{badge}</span>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


def chest_svg(kind: str, selected: str) -> str:
    pts = HEART_PTS if kind == "heart" else LUNG_PTS
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 250">',
        '<rect x="35" y="15" width="130" height="215" rx="42" fill="#f1f5f9" stroke="#94a3b8" stroke-width="2"/>',
        '<line x1="100" y1="25" x2="100" y2="215" stroke="#cbd5e1" stroke-width="2" stroke-dasharray="4 4"/>',
        '<text x="10" y="125" font-size="14" fill="#64748b" font-family="sans-serif">R</text>',
        '<text x="180" y="125" font-size="14" fill="#64748b" font-family="sans-serif">L</text>',
    ]
    for key, (x, y) in pts.items():
        on = key == selected
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="{12 if on else 8}" fill="{"#dc2626" if on else "#cbd5e1"}" '
            f'stroke="{"#7f1d1d" if on else "#94a3b8"}" stroke-width="2"/>')
    parts.append("</svg>")
    return "".join(parts)


def render_svg(svg: str, width: int = 190) -> None:
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    st.markdown(f'<img src="data:image/svg+xml;base64,{b64}" width="{width}"/>', unsafe_allow_html=True)


def stepper_html(current: int) -> str:
    items = []
    for i in range(1, 8):
        cls = "done" if i < current else ("now" if i == current else "todo")
        mark = "✓" if i < current else str(i)
        items.append(f'<div class="sx-pill {cls}"><span class="n">{mark}</span><span>{t(f"step{i}")}</span></div>')
    return '<div class="sx-stepper">' + "".join(items) + "</div>"


def current_step() -> int:
    ss = st.session_state
    if not ss.setup_done:
        return 1
    a = ss.analysis
    if a is None or a.get("error"):
        return 2
    if not a["sqi"].passed:
        return 3
    s = ss.screening
    if s is None:
        return 4
    if s["level"] == "medium" and not ss.history_done:
        return 6
    return 7


# ---------------------------------------------------------------- sidebar
def render_sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        label = st.selectbox("Language / भाषा / ಭಾಷೆ", list(LANGS), key="lang_label")
        ss.lang_code = LANGS[label]

        st.markdown("### Model status")
        for kind, spec in MODEL_SPECS.items():
            ok = spec.path.exists()
            st.write(f"{'✅' if ok else '⚠️'} `{spec.filename}` — {'found' if ok else 'not found (DEMO MODE)'}")

        if not all(model_file_present(k) for k in MODEL_SPECS):
            st.markdown("### Demo controls")
            ss.demo_scenario = st.radio(
                "Simulated outcome (demo only)", ["auto", "good", "medium", "high"],
                format_func=lambda v: {"auto": "Auto (pseudo-random)", "good": "Force GOOD",
                                       "medium": "Force MEDIUM", "high": "Force HIGH CONCERN"}[v],
                index=["auto", "good", "medium", "high"].index(ss.demo_scenario))

        st.markdown("### Local store-and-forward")
        st.metric("Pending clinician queue", pending_count())
        df = read_screenings()
        if not df.empty:
            st.dataframe(df.drop(columns=["id"]).head(8), hide_index=True, use_container_width=True)
            st.download_button("⬇️ Export all as CSV", df.to_csv(index=False).encode("utf-8"),
                               file_name="stethoedge_screenings.csv", mime="text/csv")
        if st.button("Simulate sync (demo - nothing is sent)"):
            st.toast(f"{simulate_sync()} record(s) marked as synced locally.")
            st.rerun()
        with st.expander("Delete local data"):
            st.caption(f"Stored at `data/screenings.db`. Contains no names and no audio.")
            if st.checkbox("I understand this cannot be undone", key="del_ok") and st.button("Delete all records"):
                delete_all_data()
                st.rerun()

        with st.expander("Privacy"):
            st.caption("Audio is processed in memory on this computer and is not saved or sent anywhere. "
                       "Only the screening metadata above is stored locally.")
        with st.expander("Future hardware"):
            st.caption("MEMS microphone + reference microphone → ESP32-S3 / Raspberry Pi-class processor → "
                       "offline AI → display / speaker. Not required for this software demo.")


# ---------------------------------------------------------------- header / banners
def render_header() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<p class="sx-title">🩺 StethoEdge</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sx-sub">{t("tagline")}</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="sx-disc"><b>{t("disclaimer")}</b><br>{t("proto_notice")}</div>', unsafe_allow_html=True)


def render_demo_banner() -> None:
    if not model_file_present(current_kind()):
        st.markdown(f'<div class="sx-demo"><b>{t("demo_title")}</b><br>{t("demo_sub")}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- step 1: setup
def render_setup() -> None:
    step_heading(1)
    left, right = st.columns([3, 2])
    with left:
        kind = st.radio(
            t("select_type"), ["heart", "lung"], key="kind_w", horizontal=True,
            format_func=lambda k: ("❤️ " if k == "heart" else "🫁 ") + t(k))
        position = st.radio(
            t("select_pos"), POSITIONS[kind], key=f"pos_{kind}", horizontal=True,
            format_func=lambda p: t(f"pos_{p}"))
        st.info(f"**{t('pos_' + position)}** — {POSITION_HELP[position]}")
        if st.button(t("start_btn"), type="primary"):
            st.session_state.setup_done = True
            st.rerun()
    with right:
        st.caption(t("guide_title"))
        render_svg(chest_svg(kind, position))
        st.caption(t("diagram_note"))


def render_setup_summary() -> None:
    kind, pos = current_kind(), current_position()
    c1, c2 = st.columns([4, 1])
    c1.markdown(f"**{'❤️' if kind == 'heart' else '🫁'} {t(kind)} · {t('pos_' + pos)}** — {POSITION_HELP[pos]}")
    if c2.button(t("change_setup")):
        reset_screening()
        st.rerun()


# ---------------------------------------------------------------- step 2: record
AUDIO_MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
              ".flac": "audio/flac", ".m4a": "audio/mp4"}


def render_recording():
    """Show recorder + uploader. Returns (bytes, filename) or None."""
    step_heading(2)
    st.caption(t("record_help"))
    n = st.session_state.rec_n
    data, name = None, ""
    tab_rec, tab_up = st.tabs([t("tab_record"), t("tab_upload")])
    with tab_rec:
        if hasattr(st, "audio_input"):
            f = st.audio_input(t("record_label"), key=f"rec_{n}")
            if f is not None:
                data, name = f.getvalue(), "recording.wav"
        else:
            st.warning(t("no_audio_input"))
        st.caption(t("mic_help"))
    with tab_up:
        f = st.file_uploader(t("upload_label"), type=["wav", "mp3", "flac", "ogg", "m4a"], key=f"up_{n}")
        if f is not None and data is None:
            data, name = f.getvalue(), f.name
    if data is None:
        return None
    st.caption(t("listen"))
    st.audio(data, format=AUDIO_MIME.get(Path(name).suffix.lower(), "audio/wav"))
    return data, name


def ensure_analysis(data: bytes, name: str, kind: str, position: str) -> dict:
    """Decode + SQI once per new recording; logs failed quality checks locally."""
    ss = st.session_state
    h = hashlib.sha1(data).hexdigest()
    a = ss.analysis
    if a and a["hash"] == h and a["kind"] == kind:
        return a
    ss.screening, ss.row_id, ss.history_done, ss.history_rec = None, None, False, None
    y, err = load_audio(data, name)
    a = {"hash": h, "kind": kind, "error": err, "y": y, "sqi": None}
    if y is not None:
        a["sqi"] = compute_sqi(y, TARGET_SR, kind)
        if not a["sqi"].passed:
            insert_screening(kind, position, a["sqi"], "POOR_RECORDING", None)
    ss.analysis = a
    return a


# ---------------------------------------------------------------- step 3: quality
def render_quality(a: dict) -> bool:
    step_heading(3)
    sqi: SQIResult = a["sqi"]
    st.caption(t("sqi_proto"))
    if sqi.passed:
        card("good", "✅ " + t("quality_ok"), f"{t('quality_score')}: {sqi.score:.0f} / 100")
    else:
        card("high", "⛔ " + t("poor_recording"), t("reposition"),
             " ".join(t(f"issue_{i}") for i in sqi.issues))
        if st.button(t("record_again"), type="primary"):
            reset_recording()
            st.rerun()
    with st.expander(t("quality_details")):
        st.json({"score": sqi.score, "passed": sqi.passed, "issues": sqi.issues, **sqi.metrics})
    return sqi.passed


# ---------------------------------------------------------------- steps 4-5: screening + result
def plot_waveform(res: ModelResult) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.plot(np.arange(len(res.y_proc)) / res.sr, res.y_proc, lw=0.6, color="#334155")
    ax.axvspan(res.win_start_s, res.win_start_s + res.win_s, color="#2563eb", alpha=0.15)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Filtered waveform (blue = analysed window)", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_spectrogram(res: ModelResult) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.8))
    im = ax.imshow(res.mel_db, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, res.win_s, 0, res.mel_db.shape[0]])
    ax.set_xlabel("Time within analysed window (s)")
    ax.set_ylabel("Mel bin")
    ax.set_title("Log-Mel spectrogram (dB)", fontsize=9)
    fig.colorbar(im, ax=ax, pad=0.01)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_screening(a: dict) -> None:
    ss = st.session_state
    step_heading(4)
    if ss.screening is None:
        if st.button(t("run_screening"), type="primary"):
            with st.spinner(t("screening_busy")):
                try:
                    res = run_model(current_kind(), a["y"], TARGET_SR, ss.demo_scenario)
                except Exception as exc:
                    st.error(f"Screening failed: {exc}")
                    return
            level, score = map_probabilities_to_risk(current_kind(), res.probs)
            ss.screening = {"res": res, "level": level, "score": score}
            action = {"good": "follow_local_workflow", "high": "clinical_assessment_referral"}.get(level)
            ss.row_id = insert_screening(current_kind(), current_position(), a["sqi"],
                                         {"good": "GOOD", "medium": "MEDIUM", "high": "HIGH_CONCERN"}[level],
                                         res.mode, action)
            st.rerun()
        return

    s = ss.screening
    res: ModelResult = s["res"]
    level = s["level"]
    step_heading(5)
    for note in res.notes:
        st.warning(note)
    sim = res.mode == "demo"
    if sim:
        st.markdown(f'<div class="sx-demo"><b>{t("demo_title")}</b><br>{t("demo_sub")}</div>', unsafe_allow_html=True)
    icon = {"good": "🟢", "medium": "🟡", "high": "🔴"}[level]
    card(level, f"{icon} {t(level)}", t(f"{level}_meaning"),
         badge=t("simulated_tag") if sim else "")
    st.caption(t("saved_note"))

    with st.expander(t("viz_title"), expanded=False):
        plot_waveform(res)
        plot_spectrogram(res)
        st.caption(t("viz_note"))
        st.caption(t("explain_na"))  # TODO: replace when the model can return a saliency/segment map
        if res.mode == "tflite":
            st.write({k: round(v, 3) for k, v in res.probs.items()})


# ---------------------------------------------------------------- steps 6-7: history + next action
def render_history() -> None:
    ss = st.session_state
    step_heading(6)
    n = ss.screen_n
    st.write(t("history_intro"))
    with st.form(f"history_form_{n}"):
        age = st.selectbox(t("age_group"), AGE_GROUPS, index=2)
        st.markdown(f"**{t('known_cond')}**")
        cols = st.columns(2)
        cond = {c: cols[i % 2].checkbox(t(f"cond_{c}"), key=f"cond_{n}_{c}") for i, c in enumerate(CONDITIONS)}
        tobacco = st.radio(t("tobacco"), ["no", "yes"], horizontal=True, format_func=t)
        st.markdown(f"**{t('symptoms')}**")
        cols = st.columns(3)
        sym = {k: cols[i % 3].checkbox(t(f"sym_{k}"), key=f"sym_{n}_{k}") for i, k in enumerate(SYMPTOMS)}
        submitted = st.form_submit_button(t("submit_history"), type="primary")
    if submitted:
        code = recommend_next_step({c for c, v in cond.items() if v}, tobacco == "yes",
                                   {k for k, v in sym.items() if v})
        update_history(ss.row_id, age, f"history_{code}")
        ss.history_rec, ss.history_done = code, True
        st.rerun()


def render_next_action() -> None:
    ss = st.session_state
    level = ss.screening["level"]
    step_heading(7)
    if level == "good":
        card("good", t("good_next"))
    elif level == "high":
        card("high", "⚠️ " + t("high_next"))
    else:
        code = ss.history_rec
        css = {"prompt": "high", "doctor": "medium", "protocol": "info", "none": "info"}[code]
        label = t({"prompt": "rec_prompt", "doctor": "rec_doctor",
                   "protocol": "rec_protocol", "none": "rec_none"}[code])
        small = t("rec_note")
        if code == "none":
            small = t("rec_protocol") + ". " + small
        card(css, label, small=small)
    st.markdown(f'<div class="sx-disc"><b>{t("final_note")}</b></div>', unsafe_allow_html=True)
    if st.button(t("new_screening")):
        reset_screening()
        st.rerun()


# ---------------------------------------------------------------- orchestration
def clear_analysis_if_audio_removed() -> None:
    ss = st.session_state
    if ss.analysis is not None:
        ss.analysis, ss.screening, ss.row_id = None, None, None
        ss.history_done, ss.history_rec = False, None


def run_flow() -> None:
    ss = st.session_state
    render_demo_banner()
    if not ss.setup_done:
        render_setup()
        return
    render_setup_summary()
    got = render_recording()
    if got is None:
        clear_analysis_if_audio_removed()
        return
    data, name = got
    a = ensure_analysis(data, name, current_kind(), current_position())
    if a["error"]:
        st.error(t("read_error"))
        if st.button(t("record_again"), type="primary"):
            reset_recording()
            st.rerun()
        return
    if not render_quality(a):
        return  # quality gate: no health result is ever produced for a failed recording
    render_screening(a)
    if ss.screening is None:
        return
    if ss.screening["level"] == "medium" and not ss.history_done:
        render_history()
        return
    render_next_action()


def main() -> None:
    init_state()
    init_db()
    render_sidebar()
    render_header()
    stepper_slot = st.empty()       # filled after the flow runs so it reflects the latest state
    run_flow()
    stepper_slot.markdown(stepper_html(current_step()), unsafe_allow_html=True)


main()