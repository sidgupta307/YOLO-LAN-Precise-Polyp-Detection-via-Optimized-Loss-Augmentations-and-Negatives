"""
small_polyp_ci.py

Computes a 95% bootstrap confidence interval for the small-polyp mAP50:95
subset (Table 4), using an IMAGE-LEVEL cluster bootstrap rather than naive
per-polyp resampling.

WHY IMAGE-LEVEL, NOT POLYP-LEVEL:
The small-polyp subset contains 11 polyps across only 10 test images (one
image has 2 small polyps). Those 2 polyps from the same image are not
independent -- they share that image's lighting, resolution, and model
behavior. Resampling all 11 polyps independently would slightly understate
the true uncertainty (pseudo-replication). The correct approach is to
resample the 10 IMAGES with replacement, keeping both polyps together
whenever that image is drawn, and recompute the small-polyp mAP50:95 from
whichever images/polyps end up in each resample.

METHODOLOGY MATCHES YOUR EXISTING PER-IMAGE EXTRACTION:
Reuses the same _compute_ap / _box_iou logic as your Figure 6 bootstrap and
the multiseed extraction script, restricted to ground-truth boxes below the
5% image-area threshold (Section III-F definition), matched against ALL
model predictions (so a prediction matching a non-small polyp is correctly
NOT counted as a true positive for this subset -- consistent with standard
size-stratified evaluation).

*** YOU NEED TO FILL IN WEIGHTS_PATH BELOW ***
This should be the exact YOLOv8-large full-pipeline weights used to
generate Table 4's original small-polyp row (mAP50:95 = 0.7990 per the
manuscript). If you still have that exact best.pt, point at it directly --
using a DIFFERENT trained model (e.g. one of the new multi-seed runs)
would produce a CI around a different number than the one actually
reported in Table 4, which would be inconsistent with the manuscript.

USAGE:
  python small_polyp_ci.py
"""

from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

# ============================== CONFIG ==============================
# TODO: confirm this exact path is correct on your machine -- this now
# points at YOLOv8-SMALL, matching Table 4's actual column headers
# ("YOLOv8s" / "YOLOv12s"), NOT the YOLOv8-large model used for Table 3's
# headline 0.8656. Table 4's reported small-polyp mAP50:95 = 0.799 was
# generated with the SMALL backbone -- using the large model here would
# reproduce a different number than the one actually published.
WEIGHTS_PATH = r"C:\Medical_image_analysis\yolov8_kvasir\results\YOLOv8\kvasir-seg_augmentation_10_percent_negative_yolov8\yolov8s_kvasir_train_26_06\weights\best.pt"

DATA_YAML_EVAL = r"C:\Medical_image_analysis\kvasir-seg\yolo\data.yaml"
SMALL_POLYP_AREA_THRESHOLD = 0.05  # matches Section III-F definition
N_BOOTSTRAP = 100_000
DEVICE = 0  # adjust to whichever GPU is free
# ======================================================================


def _compute_ap(tp_sorted: np.ndarray) -> float:
    if len(tp_sorted) == 0:
        return 0.0
    fp = 1.0 - tp_sorted
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(fp)
    prec = tp_cum / (tp_cum + fp_cum + 1e-9)
    rec = tp_cum / (tp_cum[-1] + 1e-9)
    rec = np.concatenate(([0.0], rec, [1.0]))
    prec = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    idx = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[idx + 1] - rec[idx]) * prec[idx + 1]))


def _box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    ix1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    iy1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    ix2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    iy2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-9)


def gather_image_level_data(model, data_yaml_eval, device):
    """
    Returns a list of per-image records, ONLY for images containing at
    least one small polyp (<5% image area). Each record holds that
    image's small-polyp ground-truth boxes and ALL model predictions
    (needed so predictions matching non-small polyps correctly do NOT
    count as true positives for the small-polyp subset).
    """
    with open(data_yaml_eval) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", "."))
    test_img_dir = root / cfg.get("test", "images/test")
    test_label_dir = Path(str(test_img_dir).replace("images", "labels"))
    image_paths = sorted(test_img_dir.glob("*.jpg")) + sorted(test_img_dir.glob("*.png"))

    records = []
    total_small_polyps = 0

    for img_path in image_paths:
        label_path = test_label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        with open(label_path) as f:
            gt_lines = [line.split() for line in f if line.strip()]
        if not gt_lines:
            continue

        result = model.predict(img_path, verbose=False, device=device)[0]
        h, w = result.orig_shape

        all_gt_boxes = []
        small_gt_boxes = []
        for p in gt_lines:
            xc, yc, bw, bh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
            area_frac = bw * bh  # normalized w*h approximates area(yi)/area(xi)
            box = [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]
            all_gt_boxes.append(box)
            if area_frac < SMALL_POLYP_AREA_THRESHOLD:
                small_gt_boxes.append(box)

        if not small_gt_boxes:
            continue  # this image has no small polyps, not part of the subset

        small_gt_boxes = np.array(small_gt_boxes, dtype=np.float64)
        total_small_polyps += len(small_gt_boxes)

        if result.boxes is not None and len(result.boxes) > 0:
            pred_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float64)
            pred_conf = result.boxes.conf.cpu().numpy()
            order = np.argsort(-pred_conf)
            pred_boxes = pred_boxes[order]
        else:
            pred_boxes = np.zeros((0, 4), dtype=np.float64)

        records.append({
            "image": img_path.name,
            "small_gt_boxes": small_gt_boxes,
            "pred_boxes": pred_boxes,
        })

    print(f"Found {len(records)} images containing small polyps, "
          f"{total_small_polyps} small polyps total.")
    if len(records) != 10 or total_small_polyps != 11:
        print(f"  WARNING: expected 10 images / 11 polyps per the manuscript's "
              f"reported n=11 across 10 images -- got {len(records)} images / "
              f"{total_small_polyps} polyps. Check SMALL_POLYP_AREA_THRESHOLD "
              f"and WEIGHTS_PATH match the original Table 4 methodology.")
    return records


def small_polyp_map_from_records(records):
    """Computes small-polyp mAP50:95 by pooling TP/FP across all given image records."""
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    aps = []
    for thr in iou_thresholds:
        all_tp = []
        for rec in records:
            gt = rec["small_gt_boxes"]
            preds = rec["pred_boxes"]
            if len(preds) == 0:
                continue
            iou_mat = _box_iou(preds, gt)
            tp = np.zeros(len(preds))
            gt_used = np.zeros(len(gt), dtype=bool)
            for pi in range(len(preds)):
                if iou_mat.shape[1] == 0:
                    break
                best_j = int(np.argmax(iou_mat[pi]))
                if iou_mat[pi, best_j] >= thr and not gt_used[best_j]:
                    tp[pi] = 1.0
                    gt_used[best_j] = True
            all_tp.append(tp)
        if all_tp:
            aps.append(_compute_ap(np.concatenate(all_tp)))
        else:
            aps.append(0.0)
    return float(np.mean(aps))


def cluster_bootstrap_ci(records, n_bootstrap=N_BOOTSTRAP, seed=42):
    """Resamples IMAGES (not individual polyps) with replacement."""
    rng = np.random.default_rng(seed)
    n_images = len(records)
    boot_scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_images, size=n_images)
        resampled = [records[j] for j in idx]
        boot_scores[i] = small_polyp_map_from_records(resampled)
    return boot_scores


def main():
    print(f"Loading model: {WEIGHTS_PATH}")
    model = YOLO(WEIGHTS_PATH)
    model.to(DEVICE)

    records = gather_image_level_data(model, DATA_YAML_EVAL, DEVICE)
    point_estimate = small_polyp_map_from_records(records)
    print(f"\nPoint estimate (small-polyp mAP50:95, all 10 images): {point_estimate:.4f}")

    print(f"Running image-level cluster bootstrap ({N_BOOTSTRAP} resamples)...")
    boot_scores = cluster_bootstrap_ci(records)
    ci_lo, ci_hi = np.percentile(boot_scores, [2.5, 97.5])

    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(f"  Small-polyp mAP50:95 = {point_estimate:.4f}")
    print(f"  95% CI (image-level cluster bootstrap) = [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Based on n=11 polyps across 10 test images (one image contains 2 polyps)")
    print("\n  Suggested manuscript sentence:")
    print(f'  "The small-polyp subset (n=11 polyps across 10 test images) achieved '
          f'mAP50:95 = {point_estimate:.4f} (95% CI [{ci_lo:.4f}, {ci_hi:.4f}], '
          f'image-level cluster bootstrap, 100,000 resamples)."')


if __name__ == "__main__":
    main()