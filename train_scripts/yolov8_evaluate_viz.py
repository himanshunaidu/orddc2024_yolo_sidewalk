import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_dataset_path(dataset_yaml: dict, value: Any, yaml_path: Path) -> List[Path]:
    """
    Resolves train/val/test entries from YOLO data.yaml.

    Supports:
    - directory of images
    - txt file containing image paths
    - list of directories/files
    """
    dataset_root = dataset_yaml.get("path", None)

    if dataset_root is None:
        base = yaml_path.parent
    else:
        dataset_root = Path(dataset_root)
        base = dataset_root if dataset_root.is_absolute() else (yaml_path.parent / dataset_root)

    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(resolve_dataset_path(dataset_yaml, v, yaml_path))
        return out

    path = Path(value)
    if not path.is_absolute():
        path = base / path

    if path.is_file() and path.suffix.lower() == ".txt":
        image_paths = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = path.parent / p
            image_paths.append(p)
        return image_paths

    if path.is_dir():
        image_paths = []
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(path.rglob(f"*{ext}"))
        return sorted(image_paths)

    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]

    raise FileNotFoundError(f"Could not resolve dataset path: {path}")


def get_image_paths_from_data_yaml(data_yaml_path: str, split: str) -> Tuple[List[Path], Dict[int, str]]:
    yaml_path = Path(data_yaml_path)
    dataset_yaml = load_yaml(str(yaml_path))

    if split not in dataset_yaml:
        raise KeyError(f"Split '{split}' not found in {data_yaml_path}. Available keys: {dataset_yaml.keys()}")

    image_paths = resolve_dataset_path(dataset_yaml, dataset_yaml[split], yaml_path)

    names = dataset_yaml.get("names", None)
    if names is None:
        raise KeyError("Dataset YAML must contain 'names'.")

    if isinstance(names, dict):
        class_names = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        class_names = {i: str(name) for i, name in enumerate(names)}
    else:
        raise ValueError("'names' must be a list or dict.")

    image_paths = [p for p in image_paths if p.suffix.lower() in IMAGE_EXTENSIONS]

    if len(image_paths) == 0:
        raise ValueError(f"No images found for split '{split}'.")

    return image_paths, class_names


def infer_label_path_from_image_path(image_path: Path) -> Path:
    """
    Standard YOLO layout:
        images/train/xxx.jpg -> labels/train/xxx.txt
        images/val/xxx.jpg   -> labels/val/xxx.txt

    If 'images' is not in the path, fallback to same directory with .txt.
    """
    parts = list(image_path.parts)

    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")

    return image_path.with_suffix(".txt")


def read_yolo_label_file(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> List[dict]:
    """
    Reads YOLO xywh-normalized labels and converts them to xyxy pixel boxes.
    Missing label file means zero objects.
    """
    if not label_path.exists():
        return []

    lines = label_path.read_text().splitlines()
    gts = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        cls = int(float(parts[0]))
        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        width = float(parts[3]) * image_width
        height = float(parts[4]) * image_height

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        x1 = max(0.0, min(float(image_width), x1))
        y1 = max(0.0, min(float(image_height), y1))
        x2 = max(0.0, min(float(image_width), x2))
        y2 = max(0.0, min(float(image_height), y2))

        if x2 <= x1 or y2 <= y1:
            continue

        gts.append({
            "class_id": cls,
            "box": np.array([x1, y1, x2, y2], dtype=np.float32),
        })

    return gts


def load_ground_truths(image_paths: List[Path]) -> Dict[str, dict]:
    """
    Returns:
        image_id -> {
            image_path,
            width,
            height,
            gts: list of {class_id, box}
        }
    """
    gt_by_image = {}

    for image_path in tqdm(image_paths, desc="Loading ground truth"):
        with Image.open(image_path) as img:
            width, height = img.size

        label_path = infer_label_path_from_image_path(image_path)

        gts = read_yolo_label_file(
            label_path=label_path,
            image_width=width,
            image_height=height,
        )

        image_id = str(image_path)

        gt_by_image[image_id] = {
            "image_path": str(image_path),
            "label_path": str(label_path),
            "width": width,
            "height": height,
            "gts": gts,
        }

    return gt_by_image


# ---------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------

def run_yolo_predictions(
    model_path: str,
    image_paths: List[Path],
    *,
    imgsz: int,
    device: str,
    conf_min: float,
    nms_iou: float,
    batch: int,
) -> Dict[str, List[dict]]:
    """
    Runs YOLO prediction once at low confidence.
    The offline evaluator later sweeps confidence thresholds.
    """
    model = YOLO(model_path)

    predictions_by_image = {}

    results_iter = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=imgsz,
        device=device,
        conf=conf_min,
        iou=nms_iou,
        batch=batch,
        stream=True,
        verbose=False,
    )

    for result in tqdm(results_iter, total=len(image_paths), desc="Running predictions"):
        image_id = str(Path(result.path))

        preds = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls in zip(boxes_xyxy, confs, classes):
                preds.append({
                    "class_id": int(cls),
                    "conf": float(conf),
                    "box": box.astype(np.float32),
                })

        predictions_by_image[image_id] = preds

    return predictions_by_image


# ---------------------------------------------------------------------
# Geometry / matching
# ---------------------------------------------------------------------

def box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def match_predictions_to_ground_truths(
    gts: List[dict],
    preds: List[dict],
    *,
    iou_threshold: float,
    class_aware: bool = True,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Greedy matching by prediction confidence.

    Returns:
        matched: list of {gt_idx, pred_idx, iou}
        unmatched_preds: list of pred dicts with pred_idx
        unmatched_gts: list of gt dicts with gt_idx
    """
    matched = []
    used_gt = set()
    used_pred = set()

    pred_order = sorted(range(len(preds)), key=lambda i: preds[i]["conf"], reverse=True)

    for pred_idx in pred_order:
        pred = preds[pred_idx]

        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt in enumerate(gts):
            if gt_idx in used_gt:
                continue

            if class_aware and pred["class_id"] != gt["class_id"]:
                continue

            iou = box_iou_xyxy(pred["box"], gt["box"])

            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None:
            used_gt.add(best_gt_idx)
            used_pred.add(pred_idx)
            matched.append({
                "gt_idx": best_gt_idx,
                "pred_idx": pred_idx,
                "iou": best_iou,
            })

    unmatched_preds = []
    for pred_idx, pred in enumerate(preds):
        if pred_idx not in used_pred:
            p = dict(pred)
            p["pred_idx"] = pred_idx
            unmatched_preds.append(p)

    unmatched_gts = []
    for gt_idx, gt in enumerate(gts):
        if gt_idx not in used_gt:
            g = dict(gt)
            g["gt_idx"] = gt_idx
            unmatched_gts.append(g)

    return matched, unmatched_preds, unmatched_gts

# ---------------------------------------------------------------------
# Visualize and save results
# ---------------------------------------------------------------------

def visualize_and_save_results(
    image_path: Path,
    gts: List[dict],
    preds: List[dict],
    matched: List[dict],
    unmatched_preds: List[dict],
    unmatched_gts: List[dict],
    class_names: Dict[int, str],
    output_dir: Path,
) -> None:
    """
    Visualizes ground truths and predictions on the image and saves the result.
    """
    from PIL import ImageDraw, ImageFont

    # Load image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Draw ground truths (green)
    for gt in gts:
        box = gt["box"]
        class_id = gt["class_id"]
        label = class_names.get(class_id, str(class_id))
        draw.rectangle(box.tolist(), outline="green", width=5)
        draw.text((box[0], box[1]), label, fill="green")

    # Draw predictions (red)
    for pred in preds:
        box = pred["box"]
        class_id = pred["class_id"]
        conf = pred["conf"]
        label = f"{class_names.get(class_id, str(class_id))} {conf:.2f}"
        draw.rectangle(box.tolist(), outline="red", width=5)
        draw.text((box[0], box[1]), label, fill="red")

    # Save the visualized image
    output_path = output_dir / image_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, help="Path to YOLO .pt model.")
    parser.add_argument("--data", type=str, required=True, help="Path to YOLO data.yaml.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=16)
    
    parser.add_argument(
        "--pred-conf-min",
        type=float,
        default=0.15
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.7,
        help="NMS IoU used by Ultralytics prediction."
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions to ground truths."
    )
    
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Custom YOLOv8 Evaluation")
    print(f"Model:      {args.model}")
    print(f"Data YAML:  {args.data}")
    print(f"Split:      {args.split}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)
    
    image_paths, class_names = get_image_paths_from_data_yaml(args.data, args.split)
    
    print(f"Found {len(image_paths)} images.")
    print(f"Classes: {class_names}")

    gt_by_image = load_ground_truths(image_paths)

    num_negative = sum(1 for r in gt_by_image.values() if len(r["gts"]) == 0)
    num_positive = len(gt_by_image) - num_negative
    num_gt_objects = sum(len(r["gts"]) for r in gt_by_image.values())

    print(f"Positive images: {num_positive}")
    print(f"Negative images: {num_negative}")
    print(f"GT objects:      {num_gt_objects}")

    predictions_by_image = run_yolo_predictions(
        model_path=args.model,
        image_paths=image_paths,
        imgsz=args.imgsz,
        device=args.device,
        conf_min=args.pred_conf_min,
        nms_iou=args.nms_iou,
        batch=args.batch,
    )
    
    # Visualize and save results
    for image_id, gt_data in gt_by_image.items():
        image_path = Path(gt_data["image_path"])
        gts = gt_data["gts"]
        preds = predictions_by_image.get(str(image_path), [])

        matched, unmatched_preds, unmatched_gts = match_predictions_to_ground_truths(
            gts=gts,
            preds=preds,
            iou_threshold=args.iou_threshold,
            class_aware=True,
        )

        visualize_and_save_results(
            image_path=image_path,
            gts=gts,
            preds=preds,
            matched=matched,
            unmatched_preds=unmatched_preds,
            unmatched_gts=unmatched_gts,
            class_names=class_names,
            output_dir=output_dir,
        )
        print(f"Processed {image_path.name}: {len(gts)} GTs, {len(preds)} preds, {len(matched)} matched.")

if __name__ == "__main__":
    main()

# export CUDA_VISIBLE_DEVICES=0   
# python yolov8_evaluate_viz.py \
# --model /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk/archive/A_full_soft_weights_yolov8x_batch_16_lr0_0.001_lrf_0.01_imgsz_640_opt_SGD_lr1e-3/weights/best.pt \
# --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
# --split val \
# --output-dir /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk/archive/A_full_soft_weights_yolov8x_batch_16_lr0_0.001_lrf_0.01_imgsz_640_opt_SGD_lr1e-3/evaluation_viz/ \
# --device 2