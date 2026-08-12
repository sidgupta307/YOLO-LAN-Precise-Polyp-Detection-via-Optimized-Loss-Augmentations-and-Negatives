"""
get_ablation_results.py

Re-runs validation on each grouped-ablation model's best.pt and prints/
saves a clean summary table (precision, recall, mAP50, mAP50:95) for
all three groups (geometric, blur, composite) in one place — no need to
scroll back through terminal output or guess at numbers from a busy log.

USAGE:
  python get_ablation_results.py
"""

from pathlib import Path
import csv

from ultralytics import YOLO

# Adjust if your ablation run folders live somewhere else
RUNS_ROOT = Path(r"C:\Medical_image_analysis\yolo-lan(9_8_2026)\runs")
GROUPS = ["geometric", "blur", "composite"]

OUT_CSV = Path(r"C:\Medical_image_analysis\yolo-lan(9_8_2026)\ablation_summary.csv")


def get_metrics_for_group(group):
    weights_path = RUNS_ROOT / f"ablation_{group}" / "weights" / "best.pt"
    if not weights_path.exists():
        print(f"WARNING: {weights_path} not found — skipping {group}")
        return None

    print(f"\n=== Validating {group} ({weights_path}) ===")
    model = YOLO(str(weights_path))
    metrics = model.val()  # uses the val split defined in that run's data.yaml

    row = {
        "group": group,
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "F1": round(
            2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-9), 4
        ),
    }
    return row


def main():
    results = []
    for group in GROUPS:
        row = get_metrics_for_group(group)
        if row is not None:
            results.append(row)

    print("\n" + "=" * 72)
    print("  GROUPED AUGMENTATION ABLATION — SUMMARY")
    print("=" * 72)
    header = f"{'Group':<12} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50:95':>10} {'F1':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['group']:<12} {r['precision']:>10.4f} {r['recall']:>10.4f} "
              f"{r['mAP50']:>10.4f} {r['mAP50_95']:>10.4f} {r['F1']:>10.4f}")

    if results:
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved to: {OUT_CSV}")
        print("\nCopy the printed table above and send it back — that's exactly "
              "what I need to write the manuscript results paragraph and table row.")


if __name__ == "__main__":
    main()