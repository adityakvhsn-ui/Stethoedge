"""Automatically sort ICBHI respiratory recordings into the project's
binary lung normal/abnormal folders using the cycle annotations.

Usage:
    python prepare_icbhi.py "C:\\path\\to\\ICBHI_final_database"
    python prepare_icbhi.py "C:\\path\\to\\ICBHI_final_database.zip"

The project's lung model uses the first 5 seconds of each recording, so this
script labels only annotated breathing cycles that overlap those first 5 sec.
A recording is NORMAL only when all overlapping annotated cycles have
crackles=0 and wheezes=0. If any overlapping cycle has either flag set to 1,
it is ABNORMAL. Files without a matching/usable annotation are skipped.
"""
from __future__ import annotations

import csv
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

MODEL_SECONDS = 5.0
PROJECT_ROOT = Path(__file__).resolve().parent
DEST_ROOT = PROJECT_ROOT / "data" / "verified_demo_lungs" / "lungs"
REPORT_PATH = PROJECT_ROOT / "icbhi_import_report.csv"
EXTRACT_DIR = PROJECT_ROOT / "data" / "ICBHI_final_database_import"


def get_source_root(argument: str) -> Path:
    source = Path(argument).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    if source.is_file() and source.suffix.lower() == ".zip":
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Extracting ZIP to: {EXTRACT_DIR}")
        with zipfile.ZipFile(source, "r") as archive:
            archive.extractall(EXTRACT_DIR)
        return EXTRACT_DIR

    if not source.is_dir():
        raise ValueError("Give this script an extracted dataset folder or a .zip file.")
    return source


def load_annotation(path: Path) -> np.ndarray | None:
    try:
        values = np.loadtxt(path, dtype=float)
        values = np.atleast_2d(values)
        if values.shape[1] < 4:
            return None
        values = values[:, :4]
        if not np.isfinite(values).all():
            return None
        return values
    except (OSError, ValueError):
        return None


def classify_first_five_seconds(annotation: np.ndarray) -> tuple[str | None, int, int]:
    # Annotation columns: cycle start, cycle end, crackles, wheezes.
    overlap_mask = (annotation[:, 0] < MODEL_SECONDS) & (annotation[:, 1] > 0)
    cycles = annotation[overlap_mask]
    if len(cycles) == 0:
        return None, 0, 0

    event_mask = (cycles[:, 2] >= 1) | (cycles[:, 3] >= 1)
    event_cycles = int(np.count_nonzero(event_mask))
    if event_cycles > 0:
        return "abnormal", len(cycles), event_cycles
    return "normal", len(cycles), 0


def main() -> int:
    if len(sys.argv) != 2:
        print('Usage: python prepare_icbhi.py "PATH_TO_ICBHI_FOLDER_OR_ZIP"')
        return 2

    source_root = get_source_root(sys.argv[1])
    wav_files = sorted(p for p in source_root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    txt_by_stem: dict[str, list[Path]] = {}
    for txt in source_root.rglob("*.txt"):
        txt_by_stem.setdefault(txt.stem.lower(), []).append(txt)

    if not wav_files:
        print(f"No WAV files found under: {source_root}")
        return 1

    for label in ("normal", "abnormal"):
        (DEST_ROOT / label).mkdir(parents=True, exist_ok=True)

    counts = {"normal": 0, "abnormal": 0}
    skipped_no_txt = 0
    skipped_bad_txt = 0
    skipped_no_cycles = 0
    skipped_existing = 0
    report_rows: list[list[object]] = []

    for wav in wav_files:
        matches = txt_by_stem.get(wav.stem.lower(), [])
        if not matches:
            skipped_no_txt += 1
            report_rows.append([wav.name, "SKIPPED", "No matching TXT annotation", str(wav)])
            continue

        # Prefer a TXT file next to its WAV when both are present.
        annotation_path = next((p for p in matches if p.parent == wav.parent), matches[0])
        annotation = load_annotation(annotation_path)
        if annotation is None:
            skipped_bad_txt += 1
            report_rows.append([wav.name, "SKIPPED", "Invalid TXT annotation", str(wav)])
            continue

        label, cycle_count, event_count = classify_first_five_seconds(annotation)
        if label is None:
            skipped_no_cycles += 1
            report_rows.append([wav.name, "SKIPPED", "No annotated cycle overlaps first 5 seconds", str(wav)])
            continue

        destination = DEST_ROOT / label / wav.name
        if destination.exists():
            skipped_existing += 1
            report_rows.append([wav.name, "EXISTS", label, str(wav)])
            continue

        shutil.copy2(wav, destination)
        counts[label] += 1
        reason = f"{cycle_count} overlapping cycle(s); {event_count} with crackles/wheezes"
        report_rows.append([wav.name, label.upper(), reason, str(wav)])

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(["filename", "assigned_folder", "reason", "source_path"])
        writer.writerows(report_rows)

    print("\n========== ICBHI IMPORT COMPLETE ==========")
    print(f"WAV files scanned       : {len(wav_files)}")
    print(f"Copied into normal      : {counts['normal']}")
    print(f"Copied into abnormal    : {counts['abnormal']}")
    print(f"Skipped (no TXT)        : {skipped_no_txt}")
    print(f"Skipped (bad TXT)       : {skipped_bad_txt}")
    print(f"Skipped (no first-5s cycles): {skipped_no_cycles}")
    print(f"Skipped (already exists): {skipped_existing}")
    print(f"Destination             : {DEST_ROOT}")
    print(f"Import report           : {REPORT_PATH}")
    print("\nNote: labels indicate annotated crackles/wheezes in the first 5 seconds, not a medical diagnosis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
