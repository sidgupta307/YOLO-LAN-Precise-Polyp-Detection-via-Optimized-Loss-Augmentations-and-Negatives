import os
import random
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ================= CONFIG =================
DATA_ROOT = "kvasir-seg"
SPLIT = "test"

MODEL_NAME = "facebook/dinov2-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Folder containing multiple cropped polyp patches
REFERENCE_PATCH_DIR = r"C:\Users\Admin\MedSAM_Project\reference_patches"

SIMILARITY_THRESH = 0.25
NMS_IOU_THRESH = 0.5
MAX_DETECTIONS = 50

# ================= SEED =================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ================= MODEL =================
def load_model():

    print("Loading DINOv2...")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    model = AutoModel.from_pretrained(MODEL_NAME)

    model.to(DEVICE)
    model.eval()

    return processor, model


# ================= BUILD MULTI-REFERENCE =================
def build_reference(processor, model):

    files = [
        f for f in os.listdir(REFERENCE_PATCH_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if len(files) == 0:
        raise ValueError("No reference patches found!")

    print(f"Using {len(files)} reference patches")

    reference_features = []

    for f in files:

        img_path = os.path.join(REFERENCE_PATCH_DIR, f)

        img = Image.open(img_path).convert("RGB")

        inputs = processor(images=img, return_tensors="pt")

        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():

            outputs = model(pixel_values)

            # =====================================================
            # IMPORTANT:
            # Use PATCH TOKENS (not CLS token)
            # =====================================================
            patch_tokens = outputs.last_hidden_state.squeeze(0)[1:]

            # Average patch tokens
            feat = patch_tokens.mean(dim=0).cpu().numpy()

        reference_features.append(feat)

    ref = np.mean(np.stack(reference_features), axis=0)

    # =========================================================
    # SAFE NORMALIZATION
    # =========================================================
    norm = np.linalg.norm(ref)

    if norm < 1e-8:
        raise ValueError("Reference feature norm near zero.")

    ref = ref / norm

    return ref


# ================= PATCH FEATURE EXTRACTION =================
def get_patch_features(processor, model, img_path):

    # =========================================================
    # INPUT VALIDATION
    # =========================================================
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)

    img = Image.open(img_path).convert("RGB")

    inputs = processor(images=img, return_tensors="pt")

    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():

        outputs = model(pixel_values)

        # patch tokens only
        patch_feats = outputs.last_hidden_state.squeeze(0)[1:]

    img_np = np.array(img)

    # =========================================================
    # TRUE PATCH GRID
    # =========================================================
    H, W = pixel_values.shape[-2:]

    patch_size = model.config.patch_size

    grid_h = H // patch_size
    grid_w = W // patch_size

    # =========================================================
    # CRITICAL SAFETY CHECK
    # =========================================================
    actual_patches = patch_feats.shape[0]

    expected_patches = grid_h * grid_w

    if expected_patches != actual_patches:

        print(f"\n[WARNING] Patch mismatch in {img_path}")
        print(f"Expected patches: {expected_patches}")
        print(f"Actual patches:   {actual_patches}")

        # safest recovery
        grid_h = int(np.sqrt(actual_patches))
        grid_w = actual_patches // grid_h

        print(f"Recovered grid: {grid_h} x {grid_w}")

        if grid_h * grid_w != actual_patches:
            raise ValueError(
                f"Unable to recover patch grid for {img_path}"
            )

    return patch_feats, img_np, grid_h, grid_w


# ================= SIMILARITY =================
def compute_similarity(patch_feats, ref):

    patch_feats = patch_feats.cpu().numpy()

    # normalize patch features
    patch_feats = patch_feats / (
        np.linalg.norm(
            patch_feats,
            axis=1,
            keepdims=True
        ) + 1e-8
    )

    similarities = np.dot(patch_feats, ref)

    return similarities


# ================= CONVERT SIMILARITY TO BOXES =================
def sim_to_boxes(similarities, h, w, grid_h, grid_w):

    sim_map = similarities.reshape(grid_h, grid_w)

    # OpenCV resize uses (width, height)
    sim_map_up = cv2.resize(sim_map, (w, h))

    mask = (sim_map_up > SIMILARITY_THRESH).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        x, y, wc, hc = cv2.boundingRect(contour)

        conf = float(
            np.max(sim_map_up[y:y+hc, x:x+wc])
        )

        boxes.append({
            'box': [x, y, x + wc, y + hc],
            'confidence': conf
        })

    return boxes


# ================= IOU =================
def iou(a, b):

    xA = max(a[0], b[0])
    yA = max(a[1], b[1])

    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])

    union = areaA + areaB - inter

    return inter / union if union > 0 else 0


# ================= NMS =================
def nms(boxes):

    if len(boxes) == 0:
        return []

    boxes = sorted(
        boxes,
        key=lambda x: x['confidence'],
        reverse=True
    )

    keep = []

    for b in boxes:

        valid = True

        for k in keep:

            if iou(b['box'], k['box']) > NMS_IOU_THRESH:
                valid = False
                break

        if valid:
            keep.append(b)

    return keep


# ================= LOAD GT =================
def load_gt(path, w, h):

    boxes = []

    # IMPORTANT:
    # return empty list for negative images
    if not os.path.exists(path):
        return boxes

    with open(path) as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            _, cx, cy, bw, bh = map(float, parts)

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            boxes.append([x1, y1, x2, y2])

    return boxes


# ================= AP =================
def compute_interpolated_ap(recalls, precisions):

    precisions = precisions.copy()

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(
            precisions[i - 1],
            precisions[i]
        )

    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))

    return np.trapz(precisions, recalls)


def compute_ap(pred_boxes, gt_boxes, iou_thresh=0.5):

    # negative image with no predictions
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0

    # negative image with predictions
    if len(gt_boxes) == 0 and len(pred_boxes) > 0:
        return 0.0

    # positive image with no predictions
    if len(gt_boxes) > 0 and len(pred_boxes) == 0:
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

            val = iou(pred['box'], gt)

            if val > best_iou:
                best_iou = val
                best_idx = j

        if best_iou >= iou_thresh:
            tp[i] = 1
            matched_gt.add(best_idx)
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / (len(gt_boxes) + 1e-8)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)

    return compute_interpolated_ap(recalls, precisions)


# ================= mAP =================
def compute_map(pred_list, gt_list, iou_thresh=0.5):

    aps = []

    for preds, gts in zip(pred_list, gt_list):
        aps.append(
            compute_ap(preds, gts, iou_thresh)
        )

    return np.mean(aps)


# ================= EVALUATION =================
def evaluate(processor, model, ref):

    img_dir = os.path.join(DATA_ROOT, "images", SPLIT)
    lbl_dir = os.path.join(DATA_ROOT, "labels", SPLIT)

    imgs = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_preds = []
    all_gts = []

    print(f"\nEvaluating on {len(imgs)} images...")

    for name in tqdm(imgs):

        base = os.path.splitext(name)[0]

        img_path = os.path.join(img_dir, name)

        gt_path = os.path.join(lbl_dir, base + ".txt")

        patch_feats, img_np, grid_h, grid_w = (
            get_patch_features(
                processor,
                model,
                img_path
            )
        )

        h, w = img_np.shape[:2]

        gt_boxes = load_gt(gt_path, w, h)

        similarities = compute_similarity(
            patch_feats,
            ref
        )

        preds = sim_to_boxes(
            similarities,
            h,
            w,
            grid_h,
            grid_w
        )

        preds = nms(preds)

        preds = preds[:MAX_DETECTIONS]

        all_preds.append(preds)
        all_gts.append(gt_boxes)

        matched_gt = set()

        for pred in preds:

            best_iou = 0
            best_idx = -1

            for j, gt in enumerate(gt_boxes):

                if j in matched_gt:
                    continue

                val = iou(pred['box'], gt)

                if val > best_iou:
                    best_iou = val
                    best_idx = j

            if best_iou >= 0.5:
                total_tp += 1
                matched_gt.add(best_idx)
            else:
                total_fp += 1

        total_fn += (
            len(gt_boxes) - len(matched_gt)
        )

    precision = total_tp / (
        total_tp + total_fp + 1e-8
    )

    recall = total_tp / (
        total_tp + total_fn + 1e-8
    )

    f1 = 2 * precision * recall / (
        precision + recall + 1e-8
    )

    map50 = compute_map(
        all_preds,
        all_gts,
        0.5
    )

    map75 = compute_map(
        all_preds,
        all_gts,
        0.75
    )

    map50_95 = np.mean([
        compute_map(all_preds, all_gts, thr)
        for thr in np.arange(0.5, 1.0, 0.05)
    ])

    print("\n" + "=" * 60)
    print("DINOv2 MULTI-REFERENCE DETECTION")
    print("=" * 60)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"mAP@0.5:   {map50:.4f}")
    #print(f"mAP@0.75:  {map75:.4f}")
    print(f"mAP@0.5:0.95: {map50_95:.4f}")

    print("=" * 60)


# ================= MAIN =================
def main():

    processor, model = load_model()

    ref = build_reference(processor, model)

    evaluate(processor, model, ref)


if __name__ == "__main__":
    main()