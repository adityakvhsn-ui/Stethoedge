import os
import shutil
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HEART_DIR = os.path.join(
    BASE_DIR,
    "data",
    "heart"
)

REFERENCE_FILE = os.path.join(
    HEART_DIR,
    "REFERENCE_combined.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "stetho_data_Heart"
)

NORMAL_DIR = os.path.join(
    OUTPUT_DIR,
    "normal"
)

ABNORMAL_DIR = os.path.join(
    OUTPUT_DIR,
    "abnormal"
)

os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(ABNORMAL_DIR, exist_ok=True)


# ============================================================
# CHECK REFERENCE FILE
# ============================================================

if not os.path.exists(REFERENCE_FILE):

    raise FileNotFoundError(
        f"REFERENCE_combined.csv not found at:\n"
        f"{REFERENCE_FILE}"
    )


# Supports both:
# normal / abnormal
# -1 / 1
# 0 / 1

NORMAL_LABELS = {
    "normal",
    "-1",
    "0"
}

ABNORMAL_LABELS = {
    "abnormal",
    "1"
}


normal_count = 0
abnormal_count = 0
missing_count = 0


# ============================================================
# READ CSV
# ============================================================

with open(
    REFERENCE_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    rows = [
        row
        for row in csv.reader(f)
        if len(row) >= 2
    ]


# ============================================================
# REMOVE HEADER IF PRESENT
# ============================================================

if rows:

    first_column = rows[0][0].strip().lower()

    if first_column in (
        "record",
        "name",
        "file"
    ):

        rows = rows[1:]


# ============================================================
# PROCESS RECORDINGS
# ============================================================

for row in rows:

    record = row[0].strip()

    classification = (
        row[1]
        .strip()
        .lower()
    )

    source_file = os.path.join(
        HEART_DIR,
        record + ".wav"
    )

    # ----------------------------
    # CHECK WAV
    # ----------------------------

    if not os.path.exists(source_file):

        print(
            f"Missing WAV: {record}.wav"
        )

        missing_count += 1

        continue


    # ----------------------------
    # NORMAL
    # ----------------------------

    if classification in NORMAL_LABELS:

        destination = os.path.join(
            NORMAL_DIR,
            record + ".wav"
        )

        normal_count += 1


    # ----------------------------
    # ABNORMAL
    # ----------------------------

    elif classification in ABNORMAL_LABELS:

        destination = os.path.join(
            ABNORMAL_DIR,
            record + ".wav"
        )

        abnormal_count += 1


    # ----------------------------
    # UNKNOWN
    # ----------------------------

    else:

        print(
            f"Unknown classification: "
            f"{record} -> {classification}"
        )

        continue


    # ----------------------------
    # COPY
    # ----------------------------

    shutil.copy2(
        source_file,
        destination
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n================================")
print("HEART DATASET SEGREGATION DONE")
print("================================")

print(
    f"Normal WAV files   : {normal_count}"
)

print(
    f"Abnormal WAV files : {abnormal_count}"
)

print(
    f"Missing WAV files  : {missing_count}"
)

print("================================")