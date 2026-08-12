"""
run_grouped_ablation.py

Full grouped augmentation ablation (geometric-only / blur-only / composite-only),
built on YOUR EXISTING pre-augmented images (no new image transforms generated),
trained with hyperparameters MATCHED EXACTLY to your main pipeline's args.yaml
(C:\\...\\yolov8l_kvasir_train_25_06\\args.yaml), so results are directly
comparable to your reported Table 8/9 numbers.

*** ASSUMES M2IOU IS ALREADY WIRED INTO YOUR LOCAL `ultralytics` IMPORT ***
(same environment you used to produce your main results — no loss-hookup
code is added here, since your env already has it.)

CONFIRMED FROM args.yaml (yolov8l_kvasir_train_25_06):
  imgsz=640, batch=16, epochs=200, patience=50, optimizer=auto, lr0=0.01
  mosaic=1.0, close_mosaic=10, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
  degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
  flipud=0.0, fliplr=0.5, erasing=0.4, auto_augment=randaugment,
  mixup=0.0, cutmix=0.0, copy_paste=0.0, seed=0, deterministic=True

STEP 1 — fill in the CONFIG block below.
STEP 2 — run this file. It will:
    a) sort your images/train1 + labels/train1 into 3 negative-ratio-balanced
       ablation folders (geometric / blur / composite)
    b) link in your existing (unaugmented) val split, identical across all 3
    c) train YOLOv8-large on each group with args.yaml-matched hyperparameters
"""

import random
import shutil
from pathlib import Path

from ultralytics import YOLO  # must resolve to YOUR M2IoU-patched install

# ============================== CONFIG ==============================
IMAGES_DIR = Path(r"C:\Medical_image_analysis\yolov8_kvasir\images\train1")
LABELS_DIR = Path(r"C:\Medical_image_analysis\yolov8_kvasir\labels\train1")

# TODO: point these at your REAL held-out val split (same one used to
# produce Table 8/9 — must be unaugmented, and identical across all 3 groups)
VAL_IMAGES_DIR = Path(r"C:\Medical_image_analysis\yolov8_kvasir\images\val")
VAL_LABELS_DIR = Path(r"C:\Medical_image_analysis\yolov8_kvasir\labels\val")

OUT_ROOT = Path(r"C:\Medical_image_analysis\yolo-lan(9_8_2026)")

NEGATIVE_RATIO = 0.10   # matches your main "10 percent negative" configuration
MODEL_WEIGHTS = "yolov8l.pt"  # matches your headline Table 8/9 model size
SEED = 0                       # matches args.yaml seed=0
MODE = "copy"                  # "copy" or "symlink" — use copy unless Developer
                                # Mode / Administrator is confirmed on this machine
# ======================================================================

SUFFIX_TO_GROUP = [
    ("_blur15_r", "composite"),
    ("_hfvfr", "geometric"),
    ("_blur15", "blur"),
    ("_hfvf", "geometric"),
    ("_hfr", "geometric"),
    ("_vfr", "geometric"),
    ("_hf", "geometric"),
    ("_vf", "geometric"),
    ("_r", "geometric"),
]
GROUPS = ["geometric", "blur", "composite"]
EXTENSIONS = [".jpg", ".jpeg", ".png"]


def classify_suffix(stem):
    for suffix, group in SUFFIX_TO_GROUP:
        if stem.endswith(suffix):
            return stem[: -len(suffix)], group
    return stem, "orig"


def is_empty_label(label_path):
    if not label_path.exists():
        return None
    return len(label_path.read_text().strip()) == 0


def place_file(src_path, dest_path, mode):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        return
    if mode == "symlink":
        dest_path.symlink_to(src_path.resolve())
    else:
        shutil.copy(src_path, dest_path)


# ---------------------------------------------------------------------
# STEP A: sort train1 into negative-ratio-balanced ablation folders
# ---------------------------------------------------------------------
def build_ablation_train_folders(images_dir, labels_dir, out_root, negative_ratio, mode, seed):
    """
    Routes EVERY image (positive or negative) by its augmentation suffix,
    exactly symmetrically:
      - no suffix ("orig")      -> goes into ALL THREE groups
      - has a group suffix      -> goes into exactly that ONE group
    This applies identically whether the image is a positive (non-empty
    label) or a negative (empty label). Negatives were augmented with the
    same suffix set as positives (confirmed: 700 negatives = 70 unique x
    10 variants), so this naturally preserves the correct ~9% negative
    ratio per group without needing artificial subsampling -- the earlier
    version of this function treated augmented-suffix negatives as
    "ambiguous" and silently dropped them, starving every group down to
    only the 70 unaugmented negatives shared across all groups.
    """
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in EXTENSIONS]
    print(f"Found {len(image_paths)} image files in {images_dir}")

    placed_counts = {g: {"positive": 0, "negative": 0} for g in GROUPS}
    missing_labels = []

    for img_path in image_paths:
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            missing_labels.append(img_path.name)
            continue

        base_id, group = classify_suffix(stem)
        empty = is_empty_label(label_path)
        kind = "negative" if empty else "positive"

        target_groups = GROUPS if group == "orig" else [group]

        for g in target_groups:
            out_img_dir = out_root / g / "images" / "train"
            out_lbl_dir = out_root / g / "labels" / "train"
            place_file(img_path, out_img_dir / img_path.name, mode)
            place_file(label_path, out_lbl_dir / label_path.name, mode)
            placed_counts[g][kind] += 1

    if missing_labels:
        print(f"WARNING: {len(missing_labels)} images skipped (no/unreadable label). "
              f"First few: {missing_labels[:5]}")

    for g in GROUPS:
        pos = placed_counts[g]["positive"]
        neg = placed_counts[g]["negative"]
        total = pos + neg
        ratio = neg / total if total else 0
        print(f"[{g}] {pos} positives + {neg} negatives = {total} total "
              f"(negative ratio = {ratio:.1%})")


# ---------------------------------------------------------------------
# STEP B: link the SAME val split into every group
# ---------------------------------------------------------------------
def link_val_split(out_root, val_images_dir, val_labels_dir, mode):
    for g in GROUPS:
        val_img_dest = out_root / g / "images" / "val"
        val_lbl_dest = out_root / g / "labels" / "val"
        for p in val_images_dir.iterdir():
            if p.suffix.lower() in EXTENSIONS:
                place_file(p, val_img_dest / p.name, mode)
        for p in val_labels_dir.iterdir():
            if p.suffix == ".txt":
                place_file(p, val_lbl_dest / p.name, mode)
    print(f"Linked val split ({len(list(val_images_dir.iterdir()))} images) into all 3 groups")


def write_data_yaml(out_root, group):
    yaml_path = out_root / group / "data.yaml"
    yaml_path.write_text(
        f"path: {out_root / group}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: polyp\n"
    )
    return yaml_path


# ---------------------------------------------------------------------
# STEP C: train each group with hyperparameters matched to args.yaml
# ---------------------------------------------------------------------
def train_group(group_name, out_root):
    data_yaml = write_data_yaml(out_root, group_name)
    print(f"\n=== Training {MODEL_WEIGHTS} on '{group_name}' augmentation only ===")

    model = YOLO(MODEL_WEIGHTS)
    model.train(
        data=str(data_yaml),
        name=f"ablation_{group_name}",
        project=str(out_root / "runs"),
        exist_ok=True,
        device=0,

        # --- exactly matched to yolov8l_kvasir_train_25_06/args.yaml ---
        imgsz=640,
        batch=16,
        epochs=200,
        patience=50,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        seed=0,
        deterministic=True,
        close_mosaic=10,
        amp=True,
        rect=False,
        cos_lr=False,
        single_cls=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,

        # --- Ultralytics default on-the-fly augmentation, LEFT ON
        #     to match the main pipeline exactly ---
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=1.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        copy_paste_mode="flip",
        auto_augment="randaugment",
        erasing=0.4,
    )
    print(f"=== Finished training: {group_name} ===")


if __name__ == "__main__":
    print("STEP A: building negative-ratio-balanced ablation folders...")
    build_ablation_train_folders(IMAGES_DIR, LABELS_DIR, OUT_ROOT, NEGATIVE_RATIO, MODE, SEED)

    print("\nSTEP B: linking val split into all 3 groups...")
    link_val_split(OUT_ROOT, VAL_IMAGES_DIR, VAL_LABELS_DIR, MODE)

    print("\nSTEP C: training all 3 ablation groups...")
    for group_name in GROUPS:
        train_group(group_name, OUT_ROOT)

    print("\nAll done. Results for each group are under:")
    print(f"  {OUT_ROOT}\\<group>\\runs\\ablation_<group>\\")
    print("Check each run's args.yaml against yolov8l_kvasir_train_25_06's args.yaml "
          "to confirm the settings matched exactly before reporting these numbers.")