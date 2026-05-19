import os
import random
import cv2
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ================= REPRODUCIBILITY =================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ================= CONFIG =================
DATA_ROOT = "kvasir-seg"
SPLIT = "test"

CHECKPOINT = "medsam_vit_b.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= MedSAM PARAMETERS =================
POINTS_PER_SIDE = 32
PRED_IOU_THRESH = 0.4
STABILITY_SCORE_THRESH = 0.8
MIN_MASK_AREA = 100

# ================= POST-PROCESSING =================
NMS_IOU_THRESH = 0.5
CONF_THRESH = 0.3
MAX_DETECTIONS = 300

# ================= AREA FILTERING =================
MIN_POLYP_AREA_RATIO = 0.0005
MAX_POLYP_AREA_RATIO = 0.5

# ================= LOAD MODEL =================
def load_medsam(checkpoint, device):

    print("Loading MedSAM...")

    model = sam_model_registry["vit_b"](
        checkpoint=checkpoint
    )

    model.to(device)
    model.eval()

    return model

# ================= MASK GENERATOR =================
def get_mask_generator(model):

    return SamAutomaticMaskGenerator(
        model=model,
        points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_SCORE_THRESH,
        min_mask_region_area=MIN_MASK_AREA,
    )

# ================= MASK → BOX =================
def mask_to_box(mask):

    y, x = np.where(mask)

    if len(x) == 0 or len(y) == 0:
        return None

    return [
        int(x.min()),
        int(y.min()),
        int(x.max()),
        int(y.max())
    ]

# ================= AREA FILTER =================
def is_valid_polyp_box(box, img_h, img_w):

    box_area = (
        (box[2] - box[0]) *
        (box[3] - box[1])
    )

    img_area = img_h * img_w

    ratio = box_area / img_area

    return (
        MIN_POLYP_AREA_RATIO <
        ratio <
        MAX_POLYP_AREA_RATIO
    )

# ================= IOU =================
def compute_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])

    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (
        (boxA[2] - boxA[0]) *
        (boxA[3] - boxA[1])
    )

    areaB = (
        (boxB[2] - boxB[0]) *
        (boxB[3] - boxB[1])
    )

    union = areaA + areaB - inter

    return inter / union if union > 0 else 0

# ================= NMS =================
def apply_nms(masks, iou_thresh=0.5):

    if not masks:
        return []

    # ==================================================
    # IMPORTANT:
    # sort by stability score
    # better proxy than predicted_iou
    # ==================================================
    masks = sorted(
        masks,
        key=lambda x: x.get(
            'stability_score',
            x['predicted_iou']
        ),
        reverse=True
    )

    kept = []
    kept_boxes = []

    for mask in masks:

        box = mask_to_box(mask['segmentation'])

        if box is None:
            continue

        keep = True

        for kept_box in kept_boxes:

            iou_val = compute_iou(
                box,
                kept_box
            )

            if iou_val > iou_thresh:
                keep = False
                break

        if keep:
            kept.append(mask)
            kept_boxes.append(box)

    return kept

# ================= LOAD GT =================
def load_gt_boxes(label_path, img_w, img_h):

    boxes = []

    # negative image
    if not os.path.exists(label_path):
        return boxes

    with open(label_path) as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            _, cx, cy, bw, bh = map(float, parts)

            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)

            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            box = [x1, y1, x2, y2]

            # ==================================================
            # IMPORTANT:
            # apply SAME area policy as predictions
            # ==================================================
            if is_valid_polyp_box(
                box,
                img_h,
                img_w
            ):
                boxes.append(box)

    return boxes

# ================= INTERPOLATED AP =================
def compute_interpolated_ap(recalls, precisions):

    # ==================================================
    # IMPORTANT:
    # avoid in-place modification
    # ==================================================
    precisions = precisions.copy()

    for i in range(
        len(precisions) - 1,
        0,
        -1
    ):

        precisions[i - 1] = max(
            precisions[i - 1],
            precisions[i]
        )

    recalls = np.concatenate(
        ([0.0], recalls, [1.0])
    )

    precisions = np.concatenate(
        ([1.0], precisions, [0.0])
    )

    ap = np.trapz(
        precisions,
        recalls
    )

    return ap

# ================= AP =================
def compute_ap(
    pred_boxes,
    gt_boxes,
    iou_thresh=0.5
):

    # negative image
    if len(gt_boxes) == 0:

        return (
            1.0
            if len(pred_boxes) == 0
            else 0.0
        )

    # no predictions
    if len(pred_boxes) == 0:
        return 0.0

    pred_boxes = sorted(
        pred_boxes,
        key=lambda x: x['confidence'],
        reverse=True
    )

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    matched_gt = set()

    for i, pred in enumerate(pred_boxes):

        best_iou = 0
        best_idx = -1

        for j, gt in enumerate(gt_boxes):

            if j in matched_gt:
                continue

            iou_val = compute_iou(
                pred['box'],
                gt
            )

            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = j

        if best_iou >= iou_thresh:

            tp[i] = 1
            matched_gt.add(best_idx)

        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / (
        len(gt_boxes) + 1e-8
    )

    precisions = tp_cum / (
        tp_cum + fp_cum + 1e-8
    )

    ap = compute_interpolated_ap(
        recalls,
        precisions
    )

    return ap

# ================= mAP =================
def compute_map(
    pred_boxes_list,
    gt_boxes_list,
    iou_thresh=0.5
):

    if len(pred_boxes_list) == 0:
        return 0.0

    aps = []

    for preds, gts in zip(
        pred_boxes_list,
        gt_boxes_list
    ):

        aps.append(
            compute_ap(
                preds,
                gts,
                iou_thresh
            )
        )

    return np.mean(aps)

# ================= MAIN EVALUATION =================
def evaluate(model, mask_generator):

    img_dir = os.path.join(
        DATA_ROOT,
        "images",
        SPLIT
    )

    lbl_dir = os.path.join(
        DATA_ROOT,
        "labels",
        SPLIT
    )

    images = [
        f for f in sorted(
            os.listdir(img_dir)
        )
        if f.lower().endswith(
            ('.jpg', '.png', '.jpeg')
        )
    ]

    if len(images) == 0:

        print(
            f"No images found in {img_dir}"
        )

        return None

    print(
        f"\nEvaluating on {len(images)} images..."
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_negative_images = 0

    all_pred_boxes = []
    all_gt_boxes = []

    all_ious = []

    for img_name in tqdm(
        images,
        desc="Processing"
    ):

        base = os.path.splitext(img_name)[0]

        img_path = os.path.join(
            img_dir,
            img_name
        )

        lbl_path = os.path.join(
            lbl_dir,
            base + ".txt"
        )

        img = cv2.imread(img_path)

        # ==================================================
        # IMPORTANT:
        # warn instead of silent skip
        # ==================================================
        if img is None:

            print(
                f"WARNING: cannot read image {img_path}"
            )

            continue

        h, w = img.shape[:2]

        gt_boxes = load_gt_boxes(
            lbl_path,
            w,
            h
        )

        # Generate masks
        masks = mask_generator.generate(img)

        # ==================================================
        # IMPORTANT:
        # confidence filtering BEFORE NMS
        # ==================================================
        filtered_masks = []

        for m in masks:

            if (
                m.get(
                    'predicted_iou',
                    0
                ) > CONF_THRESH
            ):

                filtered_masks.append(m)

        # Apply NMS
        filtered_masks = apply_nms(
            filtered_masks,
            iou_thresh=NMS_IOU_THRESH
        )

        filtered_masks = filtered_masks[
            :MAX_DETECTIONS
        ]

        # Convert masks → boxes
        pred_boxes = []

        for m in filtered_masks:

            box = mask_to_box(
                m["segmentation"]
            )

            if (
                box and
                is_valid_polyp_box(
                    box,
                    h,
                    w
                )
            ):

                pred_boxes.append({

                    'box': box,

                    # ======================================
                    # IMPORTANT:
                    # use stability score for ranking
                    # ======================================
                    'confidence': m.get(
                        'stability_score',
                        m['predicted_iou']
                    )
                })

        # ==================================================
        # IMPORTANT:
        # ALWAYS add for mAP
        # including negative images
        # ==================================================
        all_pred_boxes.append(pred_boxes)
        all_gt_boxes.append(gt_boxes)

        if len(gt_boxes) == 0:
            total_negative_images += 1

        matched_gt = set()

        for pred in pred_boxes:

            best_iou = 0
            best_idx = -1

            for j, gt_box in enumerate(gt_boxes):

                if j in matched_gt:
                    continue

                iou_val = compute_iou(
                    pred['box'],
                    gt_box
                )

                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = j

            if best_iou >= 0.5:

                total_tp += 1

                matched_gt.add(best_idx)

                all_ious.append(best_iou)

            else:
                total_fp += 1

        total_fn += (
            len(gt_boxes) -
            len(matched_gt)
        )

    # ================= FINAL METRICS =================
    precision = total_tp / (
        total_tp + total_fp + 1e-8
    )

    recall = total_tp / (
        total_tp + total_fn + 1e-8
    )

    f1 = (
        2 * precision * recall
    ) / (
        precision + recall + 1e-8
    )

    avg_iou = (
        np.mean(all_ious)
        if all_ious
        else 0
    )

    # ================= COCO METRICS =================
    map50 = compute_map(
        all_pred_boxes,
        all_gt_boxes,
        iou_thresh=0.5
    )

    map75 = compute_map(
        all_pred_boxes,
        all_gt_boxes,
        iou_thresh=0.75
    )

    map50_95 = np.mean([

        compute_map(
            all_pred_boxes,
            all_gt_boxes,
            iou_thresh=t
        )

        for t in np.arange(
            0.5,
            1.0,
            0.05
        )
    ])

    # ================= RESULTS =================
    print("\n" + "=" * 60)
    print("MEDSAM ZERO-SHOT EVALUATION")
    print("=" * 60)

    print(f"\nTotal images            : {len(images)}")
    print(f"Negative images         : {total_negative_images}")

    print(
        f"Total GT boxes          : "
        f"{total_tp + total_fn}"
    )

    print(
        f"Total predictions       : "
        f"{total_tp + total_fp}"
    )

    print(
        f"Confidence threshold    : "
        f"{CONF_THRESH}"
    )

    print(
        f"Max detections/image    : "
        f"{MAX_DETECTIONS}"
    )

    print("\nDetection Metrics:")

    print(
        f"  Precision : {precision:.4f}"
    )

    print(
        f"  Recall    : {recall:.4f}"
    )

    print(
        f"  F1-score  : {f1:.4f}"
    )

    print(
        f"  Mean IoU  : {avg_iou:.4f}"
    )

    print("\nCOCO-style Metrics:")

    print(
        f"  mAP@0.5        : {map50:.4f}"
    )

    print(
        f"  mAP@0.75       : {map75:.4f}"
    )

    print(
        f"  mAP@0.5:0.95   : {map50_95:.4f}"
    )

    print("=" * 60)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'map50': map50,
        'map75': map75,
        'map50_95': map50_95,
        'avg_iou': avg_iou
    }

# ================= MAIN =================
def main():

    print("\n" + "=" * 60)
    print("MEDSAM ZERO-SHOT POLYP DETECTION")
    print("=" * 60)

    model = load_medsam(
        CHECKPOINT,
        DEVICE
    )

    mask_generator = get_mask_generator(
        model
    )

    results = evaluate(
        model,
        mask_generator
    )

    if results:

        print("\n✅ Evaluation complete!")

        print(
            f"\nMain metric (mAP@0.5:0.95): "
            f"{results['map50_95']:.4f}"
        )

if __name__ == "__main__":
    main()