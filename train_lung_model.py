import os
import random
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
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

    # ----------------------------
    # NORMAL
    # ----------------------------

    print("\nLoading normal recordings...")

    for file_path in normal_files:

        feature = extract_mel_spectrogram(
            file_path
        )

        if feature is not None:

            X.append(feature)
            y.append(0)

    # ----------------------------
    # ABNORMAL
    # ----------------------------

    print("Loading abnormal recordings...")

    for file_path in abnormal_files:

        feature = extract_mel_spectrogram(
            file_path
        )

        if feature is not None:

            X.append(feature)
            y.append(1)

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int32
    )

    print("\nSuccessfully loaded:", len(X))

    print(
        "Normal:",
        np.sum(y == 0)
    )

    print(
        "Abnormal:",
        np.sum(y == 1)
    )

    # Add channel dimension
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

    print("\n========================================")
    print("DATA SPLIT")
    print("========================================")

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
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

    print("\nClass weights:")
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
    print("TRAINING")
    print("========================================")

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

    print("\n========================================")
    print("TEST RESULTS")
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
            target_names=[
                "normal",
                "abnormal"
            ]
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
    # SAVE MODEL
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

        print(
            "abnormal/"
        )

        print(
            "├── coarse_crackles/"
        )

        print(
            "├── fine_crackles/"
        )

        print(
            "├── pleural_rub/"
        )

        print(
            "├── rhonchi/"
        )

        print(
            "└── wheezing/"
        )

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
    # SAVE MODEL
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

    print(
        model_path
    )

    print(
        labels_path
    )


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

# Stage 1
train_binary_model()

# Stage 2
train_subtype_model()

print("\n========================================")
print("LUNG TRAINING COMPLETE")
print("========================================")