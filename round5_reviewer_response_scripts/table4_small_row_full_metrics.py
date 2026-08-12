"""
table4_small_row_full_metrics.py

Computes the COMPLETE corrected Table 4 "Small" row (Precision, Recall,
F1, mAP50, mAP50:95) for the current 12-polyp/12-image small subset,
matching Ultralytics' standard reporting convention: Precision/Recall/F1
are taken at the confidence threshold that maximizes F1 (the same
convention Ultralytics' own val() uses for its printed P/R columns),
evaluated at IoU=0.5. mAP50 and mAP50:95 are computed the same way as
your existing small_polyp_ci.py (which already gave the verified
mAP50:95 = 0.7405, CI [0.5579, 0.9182] for YOLOv8s).

Also runs the same procedure for YOLOv12s if you provide those weights
(optional -- leave YOLOV12S_WEIGHTS_PATH as None to skip).

USAGE:
  python table4_small_row_full_metrics.py
"""

from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

# ============================== CONFIG ==============================
YOLOV8S_WEIGHTS_PATH = r"C:\Medical_image_analysis\yolov8_kvasir\results\YOLOv8\kvasir-seg_augmentation_10_percent_negative_yolov8\yolov8s_kvasir_train_26_06\weights\best.pt"

# TODO: fill this in if you want the YOLOv12s row recomputed too, else leave as None
YOLOV12S_WEIGHTS_PATH = None

DATA_YAML_EVAL = r"C:\Medical_image_analysis\kvasir-seg\yolo\data.yaml"
SMALL_POLYP_AREA_THRESHOLD = 0.05
N_BOOTSTRAP = 100_000
DEVICE = 0
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


def gather_records(model, data_yaml_eval, device):
    with open(data_yaml_eval) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", "."))
    test_img_dir = root / cfg.get("test", "images/test")
    test_label_dir = Path(str(test_img_dir).replace("images", "labels"))
    image_paths = sorted(test_img_dir.glob("*.jpg")) + sorted(test_img_dir.glob("*.png"))

    records = []
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

        small_gt_boxes = []
        for p in gt_lines:
            xc, yc, bw, bh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
            if bw * bh < SMALL_POLYP_AREA_THRESHOLD:
                small_gt_boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h,
                                        (xc + bw / 2) * w, (yc + bh / 2) * h])
        if not small_gt_boxes:
            continue

        small_gt_boxes = np.array(small_gt_boxes, dtype=np.float64)

        if result.boxes is not None and len(result.boxes) > 0:
            pred_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float64)
            pred_conf = result.boxes.conf.cpu().numpy()
            order = np.argsort(-pred_conf)
            pred_boxes = pred_boxes[order]
            pred_conf = pred_conf[order]
        else:
            pred_boxes = np.zeros((0, 4), dtype=np.float64)
            pred_conf = np.zeros(0, dtype=np.float64)

        records.append({
            "image": img_path.name,
            "small_gt_boxes": small_gt_boxes,
            "pred_boxes": pred_boxes,
            "pred_conf": pred_conf,
        })

    n_polyps = sum(len(r["small_gt_boxes"]) for r in records)
    print(f"Gathered {len(records)} images, {n_polyps} small polyps.")
    return records


def compute_map(records, iou_thresholds):
    """mAP over given IoU threshold(s), pooling TP across all images."""
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
        aps.append(_compute_ap(np.concatenate(all_tp)) if all_tp else 0.0)
    return float(np.mean(aps))


def compute_precision_recall_f1_at_iou50(records):
    """
    Pools (confidence, is_tp) across all images at IoU=0.5, then finds the
    confidence threshold that maximizes F1 -- matching Ultralytics' own
    convention for its printed Precision/Recall columns.
    """
    all_conf = []
    all_tp = []
    n_gt_total = sum(len(r["small_gt_boxes"]) for r in records)

    for rec in records:
        gt = rec["small_gt_boxes"]
        preds = rec["pred_boxes"]
        conf = rec["pred_conf"]
        if len(preds) == 0:
            continue
        iou_mat = _box_iou(preds, gt)
        gt_used = np.zeros(len(gt), dtype=bool)
        for pi in range(len(preds)):
            is_tp = 0.0
            if iou_mat.shape[1] > 0:
                best_j = int(np.argmax(iou_mat[pi]))
                if iou_mat[pi, best_j] >= 0.5 and not gt_used[best_j]:
                    is_tp = 1.0
                    gt_used[best_j] = True
            all_conf.append(conf[pi])
            all_tp.append(is_tp)

    if not all_conf:
        return 0.0, 0.0, 0.0

    all_conf = np.array(all_conf)
    all_tp = np.array(all_tp)
    order = np.argsort(-all_conf)
    all_tp_sorted = all_tp[order]

    tp_cum = np.cumsum(all_tp_sorted)
    fp_cum = np.cumsum(1 - all_tp_sorted)
    precision_curve = tp_cum / (tp_cum + fp_cum + 1e-9)
    recall_curve = tp_cum / (n_gt_total + 1e-9)
    f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-9)

    best_idx = int(np.argmax(f1_curve))
    return float(precision_curve[best_idx]), float(recall_curve[best_idx]), float(f1_curve[best_idx])


def cluster_bootstrap_ci_map5095(records, n_bootstrap=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    n_images = len(records)
    boot_scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_images, size=n_images)
        resampled = [records[j] for j in idx]
        boot_scores[i] = compute_map(resampled, iou_thresholds)
    return np.percentile(boot_scores, [2.5, 97.5])


def run_for_model(name, weights_path):
    print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")
    print(f"Loading: {weights_path}")
    model = YOLO(weights_path)
    model.to(DEVICE)

    records = gather_records(model, DATA_YAML_EVAL, DEVICE)

    precision, recall, f1 = compute_precision_recall_f1_at_iou50(records)
    map50 = compute_map(records, [0.5])
    map5095 = compute_map(records, np.linspace(0.5, 0.95, 10))

    print("\nComputing 95% CI for mAP50:95 (image-level cluster bootstrap)...")
    ci_lo, ci_hi = cluster_bootstrap_ci_map5095(records)

    n_polyps = sum(len(r["small_gt_boxes"]) for r in records)
    n_images = len(records)

    print(f"\n--- Corrected Table 4 'Small' row ({name}) ---")
    print(f"  # of Polyps: {n_polyps}  (across {n_images} images)")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  mAP50:       {map50:.4f}")
    print(f"  mAP50:95:    {map5095:.4f}  (95% CI [{ci_lo:.4f}, {ci_hi:.4f}])")
    print(f"  F1 score:    {f1:.4f}")

    return {
        "n_polyps": n_polyps, "n_images": n_images,
        "precision": precision, "recall": recall,
        "map50": map50, "map5095": map5095,
        "ci_lo": ci_lo, "ci_hi": ci_hi, "f1": f1,
    }


if __name__ == "__main__":
    results = {}
    results["YOLOv8s"] = run_for_model("YOLOv8s", YOLOV8S_WEIGHTS_PATH)
    if YOLOV12S_WEIGHTS_PATH:
        results["YOLOv12s"] = run_for_model("YOLOv12s", YOLOV12S_WEIGHTS_PATH)

    print(f"\n{'=' * 70}\n  SUMMARY -- paste into Table 4\n{'=' * 70}")
    for name, r in results.items():
        print(f"{name}: n={r['n_polyps']} polyps, {r['n_images']} images | "
              f"P={r['precision']:.3f} R={r['recall']:.3f} mAP50={r['map50']:.3f} "
              f"mAP50:95={r['map5095']:.3f} (95% CI [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]) "
              f"F1={r['f1']:.3f}")
