"""
check_ablation_group_counts.py

Verifies the actual composition of each ablation group's training set by
scanning the folders directly — no reliance on old terminal output.
Reports total images, positive vs. negative counts (based on whether the
matching label file is empty), and the resulting negative ratio.

USAGE:
  python check_ablation_group_counts.py
"""

from pathlib import Path

RUNS_ROOT = Path(r"C:\Medical_image_analysis\yolo-lan(9_8_2026)")
GROUPS = ["geometric", "blur", "composite"]
EXTENSIONS = [".jpg", ".jpeg", ".png"]


def is_empty_label(label_path):
    if not label_path.exists():
        return None
    return len(label_path.read_text().strip()) == 0


def check_group(group):
    images_dir = RUNS_ROOT / group / "images" / "train"
    labels_dir = RUNS_ROOT / group / "labels" / "train"

    if not images_dir.exists():
        print(f"{group}: images/train folder not found at {images_dir}")
        return None

    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in EXTENSIONS]

    n_positive, n_negative, n_missing_label = 0, 0, 0
    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        empty = is_empty_label(label_path)
        if empty is None:
            n_missing_label += 1
        elif empty:
            n_negative += 1
        else:
            n_positive += 1

    total = n_positive + n_negative
    ratio = n_negative / total if total else 0

    return {
        "group": group,
        "total": len(image_paths),
        "positive": n_positive,
        "negative": n_negative,
        "missing_label": n_missing_label,
        "negative_ratio": ratio,
    }


def main():
    print("=" * 78)
    print("  ABLATION GROUP COMPOSITION CHECK")
    print("=" * 78)
    header = f"{'Group':<12} {'Total':>8} {'Positive':>10} {'Negative':>10} {'Neg.Ratio':>10} {'Missing':>9}"
    print(header)
    print("-" * len(header))

    for group in GROUPS:
        result = check_group(group)
        if result:
            print(f"{result['group']:<12} {result['total']:>8} {result['positive']:>10} "
                  f"{result['negative']:>10} {result['negative_ratio']:>9.1%} "
                  f"{result['missing_label']:>9}")

    print("\nExpected pattern: geometric should have ~4x more positives than "
          "blur/composite (8 augmentation variants vs. 2 each), but all three "
          "groups' negative ratios should be close to 10% -- if any group's "
          "ratio is noticeably off from the others (e.g. 30%+ on the smaller "
          "groups), that's the confound the balancing step was meant to prevent.")


if __name__ == "__main__":
    main()
