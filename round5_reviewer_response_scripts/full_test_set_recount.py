"""
full_test_set_recount.py

Reads EVERY ground-truth box in the current test set directly from the
label files (no model needed for this part) and classifies each into
small / medium / large per the manuscript's exact Section III-E / Table 4
boundaries:
    small:  area fraction < 0.05
    medium: 0.05 <= area fraction <= 0.15
    large:  area fraction > 0.15

Reports total images, total polyps, and the full small/medium/large
breakdown -- for direct comparison against Table 4's reported
11 / 37 / 53 (101 total).

USAGE:
  python full_test_set_recount.py
"""

from pathlib import Path

IMAGES_DIR = Path(r"C:\Medical_image_analysis\yolov8_kvasir\images\test")
LABELS_DIR = Path(r"C:\Medical_image_analysis\yolov8_kvasir\labels\test")
EXTENSIONS = [".jpg", ".jpeg", ".png"]

SMALL_MAX = 0.05
MEDIUM_MAX = 0.15


def classify(area_frac):
    if area_frac < SMALL_MAX:
        return "small"
    elif area_frac <= MEDIUM_MAX:
        return "medium"
    else:
        return "large"


def main():
    image_paths = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in EXTENSIONS]
    n_images = len(image_paths)

    counts = {"small": 0, "medium": 0, "large": 0}
    per_image_counts = {}  # image name -> number of boxes
    all_boxes = []  # (image_name, area_frac, category)
    missing_labels = []

    for img_path in image_paths:
        label_path = LABELS_DIR / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(img_path.name)
            per_image_counts[img_path.name] = 0
            continue

        with open(label_path) as f:
            lines = [line.split() for line in f if line.strip()]

        per_image_counts[img_path.name] = len(lines)

        for parts in lines:
            bw, bh = float(parts[3]), float(parts[4])
            area_frac = bw * bh
            category = classify(area_frac)
            counts[category] += 1
            all_boxes.append((img_path.name, area_frac, category))

    total_polyps = sum(counts.values())

    print("=" * 70)
    print("  FULL TEST-SET RECOUNT")
    print("=" * 70)
    print(f"\nTotal images found: {n_images}")
    if missing_labels:
        print(f"WARNING: {len(missing_labels)} images had no label file: {missing_labels}")

    print(f"\nTotal ground-truth polyps (all categories): {total_polyps}")
    print(f"\nBreakdown:")
    print(f"  {'Category':<10} {'Count':>8}   (manuscript Table 4 reports)")
    print(f"  {'Small':<10} {counts['small']:>8}   (11)")
    print(f"  {'Medium':<10} {counts['medium']:>8}   (37)")
    print(f"  {'Large':<10} {counts['large']:>8}   (53)")
    print(f"  {'TOTAL':<10} {total_polyps:>8}   (101)")

    # Images with more than 1 box, and what categories those boxes fall into
    multi_box_images = {name: n for name, n in per_image_counts.items() if n > 1}
    print(f"\nImages with more than one ground-truth box: {len(multi_box_images)}")
    for name in multi_box_images:
        boxes_here = [(af, cat) for (n, af, cat) in all_boxes if n == name]
        print(f"  {name}: {len(boxes_here)} boxes -> "
              + ", ".join(f"{cat} ({af:.4f})" for af, cat in boxes_here))

    # Explicit check: does the small category match the manuscript's n=11/10 claim?
    small_boxes = [(name, af) for (name, af, cat) in all_boxes if cat == "small"]
    small_images = set(name for name, af in small_boxes)
    print(f"\nSmall-polyp subset: {len(small_boxes)} polyps across {len(small_images)} images")
    if len(small_boxes) == 11 and len(small_images) == 10:
        print("  MATCHES manuscript's reported n=11 across 10 images.")
    else:
        print(f"  DOES NOT MATCH manuscript's reported n=11 across 10 images.")

    print("\n" + "=" * 70)
    print("  Full per-box detail (all 3 categories), for the record")
    print("=" * 70)
    print(f"{'Image':<40} {'Area Frac':>10} {'Category':>10}")
    for name, af, cat in sorted(all_boxes, key=lambda x: x[1]):
        print(f"{name:<40} {af:>10.4f} {cat:>10}")


if __name__ == "__main__":
    main()
