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

HEART_DIR = os.path.join(
    BASE_DIR,
    "data",
    "stetho_data_Heart"
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

        # Make exactly 5 seconds
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

        print(f"Error processing {file_path}")
        print(e)

        return None


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    X = []
    y = []

    class_names = [
        "normal",
        "abnormal"
    ]

    print("\nLoading heart dataset...\n")

    for label_index, class_name in enumerate(class_names):

        class_dir = os.path.join(
            HEART_DIR,
            class_name
        )

        if not os.path.exists(class_dir):

            print(
                f"WARNING: Folder not found: {class_dir}"
            )

            continue

        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(
                (".wav", ".mp3", ".flac")
            )
        ]

        print(
            f"{class_name}: {len(files)} recordings"
        )

        for filename in files:

            file_path = os.path.join(
                class_dir,
                filename
            )

            feature = extract_mel_spectrogram(
                file_path
            )

            if feature is not None:

                X.append(feature)
                y.append(label_index)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int32)
    )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(input_shape):

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
            2,
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
# MAIN
# ============================================================

print("\n========================================")
print("STETHOEDGE - HEART MODEL TRAINING")
print("========================================")

print(
    f"\nTensorFlow version: {tf.__version__}"
)

print(
    f"Heart data folder:\n{HEART_DIR}"
)


# ============================================================
# LOAD DATA
# ============================================================

X, y = load_dataset()


if len(X) == 0:

    raise RuntimeError(
        "\nNo heart audio files were found.\n"
        "Check:\n"
        "data/stetho_data_Heart/normal\n"
        "data/stetho_data_Heart/abnormal"
    )


print("\n========================================")
print("DATASET")
print("========================================")

print(
    f"Total recordings: {len(X)}"
)

print(
    f"Normal:   {np.sum(y == 0)}"
)

print(
    f"Abnormal: {np.sum(y == 1)}"
)


# ============================================================
# ADD CHANNEL
# ============================================================

X = X[..., np.newaxis]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

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
    f"Training: {len(X_train)}"
)

print(
    f"Testing : {len(X_test)}"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

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


# ============================================================
# BUILD MODEL
# ============================================================

model = build_model(
    input_shape=X_train.shape[1:]
)

model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("TRAINING HEART MODEL")
print("========================================")

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


history = model.fit(

    X_train,
    y_train,

    validation_split=0.20,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# TEST
# ============================================================

print("\n========================================")
print("HEART TEST RESULTS")
print("========================================")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

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


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "heart_normal_abnormal.keras"
)

model.save(model_path)


print("\n========================================")
print("HEART MODEL SAVED")
print("========================================")

print(model_path)

print("\nTraining complete.")