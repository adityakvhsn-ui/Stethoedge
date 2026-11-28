import os
import random
import re
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers, models


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LUNG_DIR = os.path.join(
    BASE_DIR,
    "data",
    "verified_demo_lungs",
    "lungs"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

SAMPLE_RATE = 16000
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

BATCH_SIZE = 32
EPOCHS = 30

SEED = 42

# False = only retrain the normal/abnormal model.
# True = also retrain the abnormal subtype model.
TRAIN_SUBTYPE_MODEL = False

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# AUDIO FEATURE EXTRACTION
# ============================================================

def extract_mel_spectrogram(file_path):

    try:

        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        # Make every recording exactly 5 seconds
        if len(audio) < SAMPLES:

            audio = np.pad(
                audio,
                (0, SAMPLES - len(audio))
            )

        else:

            audio = audio[:SAMPLES]

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        return mel_db.astype(np.float32)

    except Exception as e:

        print(f"ERROR processing: {file_path}")
        print(e)

        return None


# ============================================================
# FIND AUDIO FILES
# ============================================================

def get_audio_files(folder):

    audio_files = []

    if not os.path.exists(folder):
        return audio_files

    for root, dirs, files in os.walk(folder):

        for filename in files:

            if filename.lower().endswith(
                (".wav", ".mp3", ".flac")
            ):

                audio_files.append(
                    os.path.join(root, filename)
                )

    return audio_files


def get_patient_group(file_path):

    """Return a patient group from an ICBHI-style filename.

    ICBHI filenames begin with a numeric patient ID, for example:
    101_1b1_Al_sc_Meditron.wav. Filenames without a recognizable ID are
    treated as unique recordings and reported, because their patient identity
    cannot safely be inferred from the filename alone.
    """

    stem = os.path.splitext(os.path.basename(file_path))[0]

    # ICBHI filenames: 101_1b1_Al_sc_Meditron.wav
    match = re.match(r"^(\d+)(?:_|$)", stem)

    # Also support names such as a0001_recording.wav or patient_001.wav.
    if match is None:
        match = re.match(r"^([A-Za-z]?\d+)(?:_|$)", stem)

    if match is None:
        match = re.match(
            r"^(?:patient|subject|sub)[_-]?([A-Za-z]*\d+)(?:_|$)",
            stem,
            flags=re.IGNORECASE
        )

    if match is not None:
        return f"patient_{match.group(1).lower()}", True

    # Do not guess that two similarly named unknown recordings are the same
    # patient. Keep each path separate and clearly report this limitation.
    relative_path = os.path.relpath(file_path, LUNG_DIR)
    relative_path = relative_path.replace(os.sep, "/").lower()
    return f"unmapped_{relative_path}", False


def make_patient_level_split(y, groups, split_name, random_state):

    """Create a stratified split without placing a patient in both sets."""

    y = np.asarray(y, dtype=np.int32)
    groups = np.asarray(groups, dtype=str)

    number_of_groups = len(np.unique(groups))

    if number_of_groups < 3:
        raise RuntimeError(
            f"Not enough distinct patient groups for {split_name}. "
            f"Found {number_of_groups}; at least 3 are required."
        )

    number_of_folds = min(5, number_of_groups)

    splitter = StratifiedGroupKFold(
        n_splits=number_of_folds,
        shuffle=True,
        random_state=random_state
    )

    overall_distribution = np.bincount(y, minlength=2) / len(y)
    candidates = []

    for train_indices, held_indices in splitter.split(
        np.zeros(len(y)),
        y,
        groups
    ):

        # Both classes must be represented in both partitions.
        if len(np.unique(y[train_indices])) < 2:
            continue

        if len(np.unique(y[held_indices])) < 2:
            continue

        train_groups = set(groups[train_indices])
        held_groups = set(groups[held_indices])

        # Defensive check: this must always be empty for GroupKFold.
        if train_groups.intersection(held_groups):
            continue

        held_distribution = (
            np.bincount(y[held_indices], minlength=2) / len(held_indices)
        )

        size_error = abs(
            (len(held_indices) / len(y)) - (1.0 / number_of_folds)
        )

        class_balance_error = float(
            np.mean(np.abs(held_distribution - overall_distribution))
        )

        score = size_error + class_balance_error
        candidates.append((score, train_indices, held_indices))

    if not candidates:
        raise RuntimeError(
            f"Could not create a patient-level {split_name} split with both "
            "NORMAL and ABNORMAL samples in each partition. Check how many "
            "distinct patient IDs occur in each class and whether filenames "
            "contain patient IDs."
        )

    candidates.sort(key=lambda item: item[0])
    _, train_indices, held_indices = candidates[0]

    # Final leakage check.
    if set(groups[train_indices]).intersection(set(groups[held_indices])):
        raise RuntimeError(
            f"Patient leakage detected while creating {split_name} split."
        )

    return train_indices, held_indices


# ============================================================
# BUILD CNN
# ============================================================

def build_model(input_shape, number_of_classes):

    model = models.Sequential([

        layers.Input(shape=input_shape),

        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Dropout(0.3),

        layers.Conv2D(
            256,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.GlobalAveragePooling2D(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.4),

        layers.Dense(
            number_of_classes,
            activation="softmax"
        )
    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model


# ============================================================
# TRAIN NORMAL / ABNORMAL
# ============================================================

def train_binary_model():

    normal_dir = os.path.join(
        LUNG_DIR,
        "normal"
    )

    abnormal_dir = os.path.join(
        LUNG_DIR,
        "abnormal"
    )

    normal_files = get_audio_files(
        normal_dir
    )

    abnormal_files = get_audio_files(
        abnormal_dir
    )

    print("\n========================================")
    print("LUNG DATASET")
    print("========================================")

    print(
        f"Normal recordings   : {len(normal_files)}"
    )

    print(
        f"Abnormal recordings : {len(abnormal_files)}"
    )

    if len(normal_files) == 0:

        raise RuntimeError(
            f"No normal recordings found in:\n{normal_dir}"
        )

    if len(abnormal_files) == 0:

        raise RuntimeError(
            f"No abnormal recordings found in:\n{abnormal_dir}"
        )

    X = []
    y = []
    patient_groups = []
    recognized_patient_id = []

    # ----------------------------
    # NORMAL: CLASS INDEX 0
    # ----------------------------

    print("\nLoading normal recordings...")

    for file_path in normal_files:

        feature = extract_mel_spectrogram(
            file_path
        )

        if feature is not None:

            X.append(feature)
            y.append(0)

            patient_group, is_recognized = get_patient_group(file_path)
            patient_groups.append(patient_group)
            recognized_patient_id.append(is_recognized)

    # ----------------------------
    # ABNORMAL: CLASS INDEX 1
    # ----------------------------

    print("Loading abnormal recordings...")

    for file_path in abnormal_files:

        feature = extract_mel_spectrogram(
            file_path
        )

        if feature is not None:

            X.append(feature)
            y.append(1)

            patient_group, is_recognized = get_patient_group(file_path)
            patient_groups.append(patient_group)
            recognized_patient_id.append(is_recognized)

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int32
    )

    patient_groups = np.array(patient_groups, dtype=str)
    recognized_patient_id = np.array(recognized_patient_id, dtype=bool)

    print("\nSuccessfully loaded:", len(X))

    print(
        "Normal:",
        np.sum(y == 0)
    )

    print(
        "Abnormal:",
        np.sum(y == 1)
    )

    recognized_groups = set(patient_groups[recognized_patient_id])
    unrecognized_count = int(np.sum(~recognized_patient_id))

    print("Recognized patient IDs:", len(recognized_groups))
    print("Files without a recognizable patient ID:", unrecognized_count)

    if unrecognized_count > 0:
        print(
            "WARNING: Files without patient IDs are treated as separate "
            "recordings. If multiple such files belong to the same person, "
            "their patient relationship must be supplied separately; the "
            "script cannot infer it safely from these filenames."
        )

    if len(X) == 0:

        raise RuntimeError(
            "No usable lung recordings were loaded."
        )

    if len(np.unique(y)) != 2:

        raise RuntimeError(
            "Both normal and abnormal classes must contain usable recordings."
        )

    # Add channel dimension
    X = X[..., np.newaxis]

    # ----------------------------
    # PATIENT-LEVEL TRAIN / TEST SPLIT
    # ----------------------------

    train_full_indices, test_indices = make_patient_level_split(
        y,
        patient_groups,
        split_name="train/test",
        random_state=SEED
    )

    # Create a separate patient-level validation set from the training
    # patients. No patient can appear in train, validation, and test together.
    train_relative_indices, val_relative_indices = make_patient_level_split(
        y[train_full_indices],
        patient_groups[train_full_indices],
        split_name="train/validation",
        random_state=SEED + 1
    )

    train_indices = train_full_indices[train_relative_indices]
    val_indices = train_full_indices[val_relative_indices]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    train_patient_groups = set(patient_groups[train_indices])
    val_patient_groups = set(patient_groups[val_indices])
    test_patient_groups = set(patient_groups[test_indices])

    if (
        train_patient_groups.intersection(val_patient_groups)
        or train_patient_groups.intersection(test_patient_groups)
        or val_patient_groups.intersection(test_patient_groups)
    ):
        raise RuntimeError(
            "Patient leakage detected between train, validation, and test sets."
        )

    print("\n========================================")
    print("PATIENT-LEVEL DATA SPLIT")
    print("========================================")

    print(
        f"Training   : {len(X_train)} recordings from "
        f"{len(train_patient_groups)} patient groups"
    )
    print(
        f"Validation : {len(X_val)} recordings from "
        f"{len(val_patient_groups)} patient groups"
    )
    print(
        f"Testing    : {len(X_test)} recordings from "
        f"{len(test_patient_groups)} patient groups"
    )

    print("\nClass counts by partition (normal, abnormal):")
    print("Training  :", np.bincount(y_train, minlength=2).tolist())
    print("Validation:", np.bincount(y_val, minlength=2).tolist())
    print("Testing   :", np.bincount(y_test, minlength=2).tolist())

    print("Patient overlap check: PASSED")

    # ----------------------------
    # MODERATED CLASS WEIGHTS
    # ----------------------------

    classes = np.unique(y_train)

    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    # Square-root moderation reduces the relative
    # influence of the minority class.

    # Class 0 = NORMAL
    # Class 1 = ABNORMAL

    min_weight = np.min(raw_weights)

    class_weights = {

        int(cls): float(
            np.sqrt(weight / min_weight)
        )

        for cls, weight in zip(
            classes,
            raw_weights
        )
    }

    print("\nOriginal balanced weights:")

    print({
        int(cls): float(weight)
        for cls, weight in zip(
            classes,
            raw_weights
        )
    })

    print("\nModerated class weights:")

    print(class_weights)

    # ----------------------------
    # BUILD MODEL
    # ----------------------------

    model = build_model(
        input_shape=X_train.shape[1:],
        number_of_classes=2
    )

    print("\n========================================")
    print("LUNG NORMAL / ABNORMAL MODEL")
    print("========================================")

    print("Class mapping: 0 = NORMAL, 1 = ABNORMAL")

    model.summary()

    # ----------------------------
    # CALLBACKS
    # ----------------------------

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )
    ]

    # ----------------------------
    # TRAIN
    # ----------------------------

    print("\n========================================")
    print("TRAINING LUNG NORMAL / ABNORMAL MODEL")
    print("========================================")

    model.fit(

        X_train,
        y_train,

        validation_data=(X_val, y_val),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=callbacks,

        verbose=1
    )

    # ----------------------------
    # TEST
    # ----------------------------

    print("\n========================================")
    print("LUNG TEST RESULTS")
    print("========================================")

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(
        f"\nTest Accuracy: {accuracy * 100:.2f}%"
    )

    # ----------------------------
    # CLASSIFICATION REPORT
    # ----------------------------

    predictions = model.predict(
        X_test,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predicted_classes,
            labels=[0, 1],
            target_names=[
                "normal",
                "abnormal"
            ],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print("Rows = actual [normal, abnormal]")
    print("Columns = predicted [normal, abnormal]")

    print(
        confusion_matrix(
            y_test,
            predicted_classes,
            labels=[0, 1]
        )
    )

    # ----------------------------
    # SAVE BINARY MODEL
    # ----------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "lung_normal_abnormal.keras"
    )

    model.save(model_path)

    print("\n========================================")
    print("LUNG BINARY MODEL SAVED")
    print("========================================")

    print(model_path)


# ============================================================
# FIND ABNORMAL SUBTYPE FOLDERS
# ============================================================

def find_subtype_folders():

    abnormal_dir = os.path.join(
        LUNG_DIR,
        "abnormal"
    )

    subtype_folders = []

    if not os.path.exists(abnormal_dir):
        return subtype_folders

    for item in os.listdir(abnormal_dir):

        full_path = os.path.join(
            abnormal_dir,
            item
        )

        if os.path.isdir(full_path):

            audio_files = get_audio_files(
                full_path
            )

            if len(audio_files) > 0:

                subtype_folders.append(
                    item
                )

    return sorted(subtype_folders)


# ============================================================
# TRAIN ABNORMAL TYPE MODEL
# ============================================================

def train_subtype_model():

    subtypes = find_subtype_folders()

    print("\n========================================")
    print("CHECKING ABNORMAL SUBTYPES")
    print("========================================")

    if len(subtypes) == 0:

        print(
            "\nNo abnormal subtype folders found."
        )

        print(
            "\nExpected structure:"
        )

        print("abnormal/")
        print("├── coarse_crackles/")
        print("├── fine_crackles/")
        print("├── pleural_rub/")
        print("├── rhonchi/")
        print("└── wheezing/")

        print(
            "\nThe Normal/Abnormal model was still trained successfully."
        )

        return

    print("\nFound abnormal types:")

    for subtype in subtypes:

        print(
            f"  {subtype}"
        )

    X = []
    y = []

    # ----------------------------
    # LOAD SUBTYPE DATA
    # ----------------------------

    for label_index, subtype in enumerate(
        subtypes
    ):

        subtype_dir = os.path.join(
            LUNG_DIR,
            "abnormal",
            subtype
        )

        files = get_audio_files(
            subtype_dir
        )

        print(
            f"\n{subtype}: {len(files)} recordings"
        )

        for file_path in files:

            feature = extract_mel_spectrogram(
                file_path
            )

            if feature is not None:

                X.append(feature)
                y.append(label_index)

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int32
    )

    # ----------------------------
    # CHECK DATA
    # ----------------------------

    if len(X) == 0:

        print(
            "\nNo subtype recordings found."
        )

        return

    print("\nTotal subtype recordings:", len(X))

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    for label, count in zip(
        unique,
        counts
    ):

        print(
            f"{subtypes[label]}: {count}"
        )

    # ----------------------------
    # CHECK THAT EVERY CLASS
    # HAS ENOUGH RECORDINGS
    # ----------------------------

    if np.min(counts) < 2:

        print(
            "\nNot enough recordings in one or more subtype classes."
        )

        print(
            "Subtype model was not trained."
        )

        return

    # Add channel
    X = X[..., np.newaxis]

    # ----------------------------
    # TRAIN / TEST SPLIT
    # ----------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=SEED,

        stratify=y
    )

    # ----------------------------
    # CLASS WEIGHTS
    # ----------------------------

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = dict(
        zip(classes, weights)
    )

    # ----------------------------
    # BUILD MODEL
    # ----------------------------

    model = build_model(
        input_shape=X_train.shape[1:],
        number_of_classes=len(subtypes)
    )

    print("\n========================================")
    print("TRAINING ABNORMAL TYPE MODEL")
    print("========================================")

    model.summary()

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )
    ]

    # ----------------------------
    # TRAIN
    # ----------------------------

    model.fit(

        X_train,
        y_train,

        validation_split=0.20,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=callbacks,

        verbose=1
    )

    # ----------------------------
    # TEST
    # ----------------------------

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print("\n========================================")
    print("ABNORMAL TYPE TEST RESULTS")
    print("========================================")

    print(
        f"\nTest Accuracy: {accuracy * 100:.2f}%"
    )

    predictions = model.predict(
        X_test,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predicted_classes,
            target_names=subtypes
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predicted_classes
        )
    )

    # ----------------------------
    # SAVE SUBTYPE MODEL
    # ----------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "lung_abnormal_type.keras"
    )

    model.save(model_path)

    # Save subtype names
    labels_path = os.path.join(
        MODEL_DIR,
        "lung_abnormal_labels.npy"
    )

    np.save(
        labels_path,
        np.array(subtypes)
    )

    print("\n========================================")
    print("ABNORMAL TYPE MODEL SAVED")
    print("========================================")

    print(model_path)
    print(labels_path)


# ============================================================
# MAIN
# ============================================================

print("\n========================================")
print("STETHOEDGE LUNG MODEL TRAINING")
print("========================================")

print(
    f"TensorFlow version: {tf.__version__}"
)

print(
    f"Lung data folder:\n{LUNG_DIR}"
)

# Stage 1: retrain the normal/abnormal lung model.
train_binary_model()

# Stage 2 is optional. Disabled by default to preserve
# the existing subtype model and labels.
if TRAIN_SUBTYPE_MODEL:

    train_subtype_model()

else:

    print(
        "\nSkipping subtype retraining; existing subtype "
        "model and labels are preserved."
    )

print("\n========================================")
print("LUNG TRAINING COMPLETE")
print("========================================")