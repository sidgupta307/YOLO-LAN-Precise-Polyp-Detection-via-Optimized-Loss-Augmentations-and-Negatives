"""
multiseed_statistical_test.py  (v3 — CLI-driven, dual-GPU safe)

Addresses: "Bootstrap resampling of predictions from one trained model
captures test-set variability, not variability from initialization or
stochastic training. The main configurations should be trained multiple
times, with means, uncertainty, and appropriate statistical comparisons
reported. Extremely precise p-values from only 100 test images should
also be interpreted cautiously."

*** WHY THIS VERSION IS DIFFERENT FROM BEFORE ***

1. DATASET PATHS FIXED: your CIoU baseline and full-pipeline configs use
   TWO DIFFERENT dataset YAMLs in your actual args.yaml files, not one
   shared YAML. This version uses the correct one per config:
     - ciou          -> C:\\Medical_image_analysis\\kvasir-seg\\yolo\\data.yaml
     - full_pipeline -> C:\\Medical_image_analysis\\yolov8_kvasir\\kvasir.yaml

2. NO MORE IN-FILE CONFIG EDITING. Every setting that previously required
   opening the file and changing a constant (RUN_MODE, ACTIVE_CONFIGS,
   DEVICE) is now a command-line argument. This eliminates the entire
   class of error we hit repeatedly before (edited-but-not-saved,
   stale-process-still-running-old-code, etc.) because there is nothing
   to edit -- you just pass different flags to the same file.

3. SAFE FOR TRUE PARALLEL EXECUTION: run CIoU on GPU 0 in the `ciou` env
   and full_pipeline on GPU 1 in the `M2IoU` env AT THE SAME TIME, in two
   separate terminals. They write to different subfolders and different
   .npy filenames, so there is no file-collision risk running both at once.

=================================== USAGE ===================================

TERMINAL A (ciou environment, GPU 0):
    conda activate ciou
    cd C:\\Medical_image_analysis\\yolo-lan(9_8_2026)
    python multiseed_statistical_test.py --run_mode all --config ciou --device 0

TERMINAL B (M2IoU environment, GPU 1) -- run this AT THE SAME TIME as Terminal A:
    conda activate M2IoU
    cd C:\\Medical_image_analysis\\yolo-lan(9_8_2026)
    python multiseed_statistical_test.py --run_mode all --config full_pipeline --device 1

AFTER BOTH TERMINALS FINISH (either environment, needs numpy+scipy only):
    python multiseed_statistical_test.py --run_mode analyse

===============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

# ============================== FIXED PATHS ==============================
# These do NOT change between runs -- only CLI args change per invocation.

DATA_YAML_CIOU = r"C:\Medical_image_analysis\kvasir-seg\yolo\data.yaml"
DATA_YAML_FULL_PIPELINE = r"C:\Medical_image_analysis\yolo-lan(9_8_2026)\full_pipeline_data.yaml"
DATA_YAML_EVAL = r"C:\Medical_image_analysis\kvasir-seg\yolo\data.yaml"  # held-out test split, same for both configs

OUT_ROOT = Path(r"C:\Medical_image_analysis\yolo-lan(9_8_2026)\bootstrap")
NPY_DIR = OUT_ROOT / "npy_scores"

SEEDS = [0, 1, 2]
MODEL_WEIGHTS_BASE = "yolov8l.pt"

CONFIGS = {
    "ciou": {
        "display": "CIoU baseline",
        "data": DATA_YAML_CIOU,
    },
    "full_pipeline": {
        "display": "M2IoU + Aug + 10% Neg (full pipeline)",
        "data": DATA_YAML_FULL_PIPELINE,
    },
}
# ===========================================================================


def pick_free_gpu(exclude=None, override=None):
    import torch

    if override is not None:
        print(f"[GPU] Using explicitly specified device: cuda:{override}")
        return override

    exclude = exclude or []
    n_devices = torch.cuda.device_count()
    if n_devices == 0:
        print("[GPU] No CUDA devices found -- falling back to CPU")
        return "cpu"

    free_mem = {}
    for i in range(n_devices):
        free, total = torch.cuda.mem_get_info(i)
        free_mem[i] = free
        print(f"[GPU] cuda:{i} -- {free / 1e9:.2f} GB free / {total / 1e9:.2f} GB total"
              f"{'  (excluded)' if i in exclude else ''}")

    candidates = {i: m for i, m in free_mem.items() if i not in exclude}
    if not candidates:
        candidates = free_mem
    chosen = max(candidates, key=candidates.get)
    print(f"[GPU] Selected cuda:{chosen} ({free_mem[chosen] / 1e9:.2f} GB free)\n")
    return chosen


def train_seeds(config_key, device, workers):
    from ultralytics import YOLO  # must resolve to the correct env's ultralytics for this config

    config = CONFIGS[config_key]
    print(f"Dataset for '{config_key}': {config['data']}")

    for seed in SEEDS:
        run_name = f"{config_key}_seed{seed}"
        print(f"\n=== Training {run_name} on cuda:{device} ===")

        model = YOLO(MODEL_WEIGHTS_BASE)
        model.train(
            data=config["data"],
            name=run_name,
            project=str(OUT_ROOT / "runs"),
            exist_ok=True,
            device=device,

            # --- exactly matched to your original args.yaml files ---
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
            seed=seed,
            deterministic=True,
            close_mosaic=10,
            amp=True,
            rect=False,
            cos_lr=False,
            single_cls=False,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            workers=workers,

            # --- default Ultralytics on-the-fly augmentation, LEFT ON to match main pipeline ---
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
        print(f"=== Finished: {run_name} ===")

    print(f"\nAll seeds complete for '{config_key}'. Weights under:")
    print(f"  {OUT_ROOT / 'runs'}\\{config_key}_seed<N>\\weights\\best.pt")


# ---------------------------------------------------------------------
# Per-image mAP50:95 extraction (same method as your original notebook)
# ---------------------------------------------------------------------
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
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    ix1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    iy1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    ix2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    iy2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-9)


def extract_per_image_map(weights_path, out_npy_path, device):
    import yaml
    from ultralytics import YOLO

    iou_thresholds = np.linspace(0.5, 0.95, 10)
    model = YOLO(weights_path)
    model.to(device if device != "cpu" else "cpu")

    with open(DATA_YAML_EVAL) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", "."))
    test_img_dir = root / cfg.get("test", "images/test")
    test_label_dir = Path(str(test_img_dir).replace("images", "labels"))
    image_paths = sorted(test_img_dir.glob("*.jpg")) + sorted(test_img_dir.glob("*.png"))

    print(f"Extracting per-image mAP50:95 for {weights_path} ({len(image_paths)} images)...")

    per_image_map = []
    for idx, img_path in enumerate(image_paths, 1):
        print(f"  [{idx:>3}/{len(image_paths)}] {img_path.name}", end="\r")
        label_path = test_label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            per_image_map.append(0.0)
            continue
        with open(label_path) as f:
            gt_lines = [line.split() for line in f if line.strip()]
        if not gt_lines:
            per_image_map.append(0.0)
            continue

        result = model.predict(img_path, verbose=False, device=device)[0]
        h, w = result.orig_shape
        gt_boxes = np.array([
            [
                (float(p[1]) - float(p[3]) / 2) * w,
                (float(p[2]) - float(p[4]) / 2) * h,
                (float(p[1]) + float(p[3]) / 2) * w,
                (float(p[2]) + float(p[4]) / 2) * h,
            ]
            for p in gt_lines
        ], dtype=np.float64)

        if result.boxes is None or len(result.boxes) == 0:
            per_image_map.append(0.0)
            continue

        pred_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float64)
        pred_conf = result.boxes.conf.cpu().numpy()
        pred_boxes = pred_boxes[np.argsort(-pred_conf)]
        iou_mat = _box_iou(pred_boxes, gt_boxes)

        aps = []
        for thr in iou_thresholds:
            tp = np.zeros(len(pred_boxes), dtype=np.float64)
            gt_used = np.zeros(len(gt_boxes), dtype=bool)
            for pi in range(len(pred_boxes)):
                best_j = int(np.argmax(iou_mat[pi]))
                if iou_mat[pi, best_j] >= thr and not gt_used[best_j]:
                    tp[pi] = 1.0
                    gt_used[best_j] = True
            aps.append(_compute_ap(tp))
        per_image_map.append(float(np.mean(aps)))

    arr = np.array(per_image_map, dtype=np.float64)
    np.save(out_npy_path, arr)
    print(f"\n  Saved {len(arr)} scores -> {out_npy_path}  "
          f"(mean={arr.mean():.4f}, std={arr.std():.4f})")
    return arr


def extract_seeds(config_key, device):
    NPY_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        run_name = f"{config_key}_seed{seed}"
        weights_path = OUT_ROOT / "runs" / run_name / "weights" / "best.pt"
        out_npy = NPY_DIR / f"{run_name}.npy"
        if not weights_path.exists():
            print(f"WARNING: {weights_path} not found -- skipping {run_name}")
            continue
        extract_per_image_map(str(weights_path), out_npy, device)


# ---------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------
def load_seed_scores(config_key):
    arrays = []
    for seed in SEEDS:
        path = NPY_DIR / f"{config_key}_seed{seed}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path} -- run extract for this config first")
        arrays.append(np.load(path))
    return arrays


def hierarchical_bootstrap_pvalue(seed_scores_a, seed_scores_b, n_bootstrap=100_000, seed=42):
    """
    Two-stage bootstrap: resample WHICH seed, then resample images WITHIN
    that seed's per-image scores. Builds the bootstrap sampling distribution
    of (mean_b - mean_a), then tests whether that distribution is consistent
    with a TRUE DIFFERENCE OF ZERO (standard percentile bootstrap hypothesis
    test) -- NOT whether resampled differences exceed the observed difference,
    which is a self-referential comparison that returns ~0.5 regardless of
    whether a real effect exists, since the observed difference sits at the
    center of its own bootstrap distribution by construction.
    """
    rng = np.random.default_rng(seed)
    n_seeds_a = len(seed_scores_a)
    n_seeds_b = len(seed_scores_b)

    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        chosen_a = rng.integers(0, n_seeds_a, size=n_seeds_a)
        chosen_b = rng.integers(0, n_seeds_b, size=n_seeds_b)
        boot_means_a = [rng.choice(seed_scores_a[si], size=len(seed_scores_a[si]), replace=True).mean()
                         for si in chosen_a]
        boot_means_b = [rng.choice(seed_scores_b[si], size=len(seed_scores_b[si]), replace=True).mean()
                         for si in chosen_b]
        boot_diffs[i] = np.mean(boot_means_b) - np.mean(boot_means_a)

    observed_diff = (np.mean([s.mean() for s in seed_scores_b]) -
                      np.mean([s.mean() for s in seed_scores_a]))

    # Two-sided percentile bootstrap test: what fraction of the bootstrap
    # distribution falls on the OPPOSITE side of zero from the observed
    # effect? Doubled for a two-sided test.
    if observed_diff >= 0:
        p_one_sided = np.mean(boot_diffs <= 0)
    else:
        p_one_sided = np.mean(boot_diffs >= 0)
    p = min(1.0, 2 * p_one_sided)

    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

    p_display = max(p, 0.001) if p < 0.001 else p
    return p, p_display, ci_lo, ci_hi


def run_analysis():
    print("=" * 72)
    print("  MULTI-SEED TRAINING-VARIABILITY ANALYSIS")
    print("=" * 72)

    scores = {}
    for config_key in CONFIGS:
        try:
            scores[config_key] = load_seed_scores(config_key)
        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            print("Both 'ciou' and 'full_pipeline' must have completed train+extract "
                  "before running analyse. Aborting.")
            return

    print("\n--- (a) Per-seed results (training-run variability) ---\n")
    seed_level_means = {}
    for config_key, config in CONFIGS.items():
        per_seed_means = [arr.mean() for arr in scores[config_key]]
        seed_level_means[config_key] = per_seed_means
        overall_mean = np.mean(per_seed_means)
        overall_std = np.std(per_seed_means, ddof=1)
        print(f"{config['display']}:")
        for seed, m in zip(SEEDS, per_seed_means):
            print(f"    seed {seed}: mAP50:95 = {m:.4f}")
        print(f"    -> mean across {len(SEEDS)} seeds = {overall_mean:.4f} +/- {overall_std:.4f} (std)\n")

    print("--- (b) Paired comparison across seed-level means (low power, n=3) ---\n")
    a_means = seed_level_means["ciou"]
    b_means = seed_level_means["full_pipeline"]
    t_stat, t_p = stats.ttest_rel(b_means, a_means)
    print(f"Paired t-test (full pipeline vs. CIoU baseline): t = {t_stat:.4f}, p = {t_p:.4f}")
    print("  CAVEAT: n=3 seeds gives very low statistical power; reported for completeness.\n")

    print("--- (c) Hierarchical (seed + image) bootstrap ---\n")
    p_exact, p_display, ci_lo, ci_hi = hierarchical_bootstrap_pvalue(scores["ciou"], scores["full_pipeline"])
    print(f"Hierarchical bootstrap: mean difference (full pipeline - CIoU) 95% CI = "
          f"[{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Hierarchical bootstrap p-value (two-sided, tests difference against zero): "
          f"p {'<' if p_exact < 0.001 else '='} {p_display:.4f}")
    print("  Precision intentionally capped at 3 decimals -- only 3 training runs "
          "contribute to the seed-level variance component.\n")

    print("=" * 72)
    print("  Suggested manuscript language:")
    print("=" * 72)
    for config_key, config in CONFIGS.items():
        m = np.mean(seed_level_means[config_key])
        s = np.std(seed_level_means[config_key], ddof=1)
        print(f"  {config['display']}: mAP50:95 = {m:.4f} +/- {s:.4f} (mean +/- std across 3 runs)")
    print(f"  Hierarchical bootstrap 95% CI for the difference: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Hierarchical bootstrap p-value: p {'<' if p_exact < 0.001 else '='} {p_display:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", choices=["train", "extract", "all", "analyse"], required=True)
    parser.add_argument("--config", choices=list(CONFIGS.keys()),
                         help="Required for train/extract/all. Not used for analyse.")
    parser.add_argument("--device", type=int, default=None,
                         help="GPU index to use. If omitted, auto-picks the GPU with most free memory.")
    parser.add_argument("--workers", type=int, default=4,
                         help="Dataloader workers, matches original args.yaml default of 4.")
    args = parser.parse_args()

    if args.run_mode == "analyse":
        run_analysis()
    else:
        if not args.config:
            raise SystemExit("ERROR: --config is required for --run_mode train/extract/all "
                              "(choose 'ciou' or 'full_pipeline')")
        device = pick_free_gpu(override=args.device)

        if args.run_mode == "train":
            train_seeds(args.config, device, args.workers)
        elif args.run_mode == "extract":
            extract_seeds(args.config, device)
        elif args.run_mode == "all":
            train_seeds(args.config, device, args.workers)
            extract_seeds(args.config, device)
            print(f"\n'{args.config}' complete. Run --run_mode analyse once BOTH configs are done.")