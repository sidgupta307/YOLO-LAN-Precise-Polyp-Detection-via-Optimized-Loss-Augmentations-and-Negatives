"""
check_negative_pool.py

Cross-checks how many negative images actually exist in your source
images/train1 + labels/train1 folder, using TWO independent methods:

  Method A: label file is empty (the heuristic used by the filtering
            script for the grouped ablation)
  Method B: filename contains a negative-sample naming pattern
            (e.g. "_neg_", based on the naming seen earlier:
            seq23_seq23_neg_0076, seq22_seq22_neg_00182)

If these two methods disagree substantially, that tells us the
empty-label heuristic is missing negatives that actually have some
non-empty placeholder content in their label file -- which would mean
the ablation groups are negative-starved not because negatives don't
exist, but because they weren't being detected correctly.

USAGE:
  python check_negative_pool.py
"""

from pathlib import Path

IMAGES_DIR = Path(r"C:\Medical_image_analysis\yolov8_kvasir\images\train1")
LABELS_DIR = Path(r"C:\Medical_image_analysis\yolov8_kvasir\labels\train1")
EXTENSIONS = [".jpg", ".jpeg", ".png"]

# Adjust this if your negative images use a different naming convention
NEGATIVE_FILENAME_MARKER = "_neg_"


def main():
    image_paths = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in EXTENSIONS]
    print(f"Total images in {IMAGES_DIR}: {len(image_paths)}\n")

    empty_label_negatives = []
    filename_marker_negatives = []
    disagreements = []

    for img_path in image_paths:
        label_path = LABELS_DIR / f"{img_path.stem}.txt"

        is_empty_by_label = label_path.exists() and len(label_path.read_text().strip()) == 0
        is_negative_by_name = NEGATIVE_FILENAME_MARKER in img_path.stem.lower()

        if is_empty_by_label:
            empty_label_negatives.append(img_path.name)
        if is_negative_by_name:
            filename_marker_negatives.append(img_path.name)

        if is_empty_by_label != is_negative_by_name:
            label_content = label_path.read_text().strip() if label_path.exists() else "<NO LABEL FILE>"
            disagreements.append((img_path.name, is_empty_by_label, is_negative_by_name, label_content[:80]))

    print(f"Method A (empty label file):        {len(empty_label_negatives)} negatives found")
    print(f"Method B (filename contains '_neg_'): {len(filename_marker_negatives)} negatives found")

    if len(empty_label_negatives) == len(filename_marker_negatives):
        print("\nBoth methods agree on the COUNT. ", end="")
    else:
        print(f"\nMETHODS DISAGREE on count by {abs(len(empty_label_negatives) - len(filename_marker_negatives))}. ")

    if disagreements:
        print(f"\n{len(disagreements)} individual files where the two methods disagree "
              f"(first 15 shown):")
        print(f"{'Filename':<45} {'EmptyLabel':>11} {'NameMarker':>11}  Label content")
        for name, a, b, content in disagreements[:15]:
            print(f"{name:<45} {str(a):>11} {str(b):>11}  {content}")
        print("\nIf 'EmptyLabel=False, NameMarker=True' shows up a lot, those images "
              "have '_neg_' in their filename but a NON-EMPTY label file — meaning "
              "either they aren't true negatives, or their label file contains a "
              "placeholder that isn't actually empty (e.g. a comment line, or "
              "whitespace not caught by .strip()). Worth opening a couple of these "
              "label files manually to see what's actually in them.")
    else:
        print("No disagreements — every file both methods flagged is consistent.")

    print(f"\n--- Implication for the 10% negative ratio ---")
    n_pos_estimate = len(image_paths) - len(empty_label_negatives)
    ratio = len(empty_label_negatives) / len(image_paths) if image_paths else 0
    print(f"If Method A's {len(empty_label_negatives)} negatives is correct: "
          f"negative ratio in train1 = {ratio:.1%} (out of {len(image_paths)} total images)")
    print(f"For a true 10% ratio with {n_pos_estimate} positives, you would need "
          f"~{round(n_pos_estimate * 0.10 / 0.90)} negatives — "
          f"{'matches' if abs(round(n_pos_estimate*0.10/0.90) - len(empty_label_negatives)) < 20 else 'does NOT match'} "
          f"what was actually found.")


if __name__ == "__main__":
    main()