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
        parts[idx] = "labels_2"
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
# AP / mAP
# ---------------------------------------------------------------------

def compute_ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    VOC/COCO-style integral over the precision envelope.
    """
    if recall.size == 0:
        return np.nan

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])

    return float(ap)


def compute_ap_for_class_at_iou(
    gt_by_image: Dict[str, dict],
    predictions_by_image: Dict[str, List[dict]],
    *,
    class_id: int,
    iou_threshold: float,
) -> dict:
    """
    Computes AP for one class at one IoU threshold.
    """
    num_gt = 0

    gt_used_by_image = {}

    for image_id, record in gt_by_image.items():
        gt_indices = [
            idx for idx, gt in enumerate(record["gts"])
            if gt["class_id"] == class_id
        ]
        num_gt += len(gt_indices)
        gt_used_by_image[image_id] = set()

    if num_gt == 0:
        return {
            "class_id": class_id,
            "iou_threshold": iou_threshold,
            "num_gt": 0,
            "ap": np.nan,
        }

    all_preds = []

    for image_id, preds in predictions_by_image.items():
        for pred_idx, pred in enumerate(preds):
            if pred["class_id"] == class_id:
                all_preds.append({
                    "image_id": image_id,
                    "pred_idx": pred_idx,
                    "conf": pred["conf"],
                    "box": pred["box"],
                })

    all_preds.sort(key=lambda x: x["conf"], reverse=True)

    tp = np.zeros(len(all_preds), dtype=np.float32)
    fp = np.zeros(len(all_preds), dtype=np.float32)

    for k, pred in enumerate(all_preds):
        image_id = pred["image_id"]

        gt_record = gt_by_image[image_id]
        candidate_gts = [
            (idx, gt) for idx, gt in enumerate(gt_record["gts"])
            if gt["class_id"] == class_id
        ]

        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt in candidate_gts:
            if gt_idx in gt_used_by_image[image_id]:
                continue

            iou = box_iou_xyxy(pred["box"], gt["box"])

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            tp[k] = 1.0
            gt_used_by_image[image_id].add(best_gt_idx)
        else:
            fp[k] = 1.0

    if len(all_preds) == 0:
        return {
            "class_id": class_id,
            "iou_threshold": iou_threshold,
            "num_gt": num_gt,
            "num_predictions": 0,
            "ap": 0.0,
        }

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recall = cum_tp / max(num_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    ap = compute_ap_from_pr(recall, precision)

    return {
        "class_id": class_id,
        "iou_threshold": iou_threshold,
        "num_gt": int(num_gt),
        "num_predictions": int(len(all_preds)),
        "ap": float(ap),
    }


def compute_map_table(
    gt_by_image: Dict[str, dict],
    predictions_by_image: Dict[str, List[dict]],
    class_names: Dict[int, str],
    iou_thresholds: List[float],
) -> pd.DataFrame:
    rows = []

    for class_id, class_name in class_names.items():
        for iou_thr in iou_thresholds:
            row = compute_ap_for_class_at_iou(
                gt_by_image,
                predictions_by_image,
                class_id=class_id,
                iou_threshold=iou_thr,
            )
            row["class_name"] = class_name
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Fixed-threshold metrics
# ---------------------------------------------------------------------

def compute_object_metrics_at_threshold(
    gt_by_image: Dict[str, dict],
    predictions_by_image: Dict[str, List[dict]],
    class_names: Dict[int, str],
    *,
    conf_threshold: float,
    iou_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        per_class_metrics_df
        per_image_metrics_df
        error_details_df
    """
    class_ids = sorted(class_names.keys())

    counts = {
        c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
        for c in class_ids
    }

    per_image_rows = []
    error_rows = []

    for image_id, gt_record in gt_by_image.items():
        gts = gt_record["gts"]

        preds = [
            p for p in predictions_by_image.get(image_id, [])
            if p["conf"] >= conf_threshold
        ]

        matched, unmatched_preds, unmatched_gts = match_predictions_to_ground_truths(
            gts=gts,
            preds=preds,
            iou_threshold=iou_threshold,
            class_aware=True,
        )

        image_tp = len(matched)
        image_fp = len(unmatched_preds)
        image_fn = len(unmatched_gts)

        for gt in gts:
            counts[gt["class_id"]]["gt"] += 1

        for pred in preds:
            counts[pred["class_id"]]["pred"] += 1

        for m in matched:
            gt = gts[m["gt_idx"]]
            c = gt["class_id"]
            counts[c]["tp"] += 1

        for pred in unmatched_preds:
            c = pred["class_id"]
            counts[c]["fp"] += 1

            error_rows.append({
                "image_id": image_id,
                "image_path": gt_record["image_path"],
                "error_type": "false_positive",
                "class_id": c,
                "class_name": class_names.get(c, str(c)),
                "confidence": pred["conf"],
                "x1": pred["box"][0],
                "y1": pred["box"][1],
                "x2": pred["box"][2],
                "y2": pred["box"][3],
            })

        for gt in unmatched_gts:
            c = gt["class_id"]
            counts[c]["fn"] += 1

            error_rows.append({
                "image_id": image_id,
                "image_path": gt_record["image_path"],
                "error_type": "false_negative",
                "class_id": c,
                "class_name": class_names.get(c, str(c)),
                "confidence": np.nan,
                "x1": gt["box"][0],
                "y1": gt["box"][1],
                "x2": gt["box"][2],
                "y2": gt["box"][3],
            })

        per_image_rows.append({
            "image_id": image_id,
            "image_path": gt_record["image_path"],
            "num_gt": len(gts),
            "num_pred": len(preds),
            "tp": image_tp,
            "fp": image_fp,
            "fn": image_fn,
            "is_negative_image": len(gts) == 0,
            "has_prediction": len(preds) > 0,
        })

    rows = []

    for c in class_ids:
        tp = counts[c]["tp"]
        fp = counts[c]["fp"]
        fn = counts[c]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        rows.append({
            "class_id": c,
            "class_name": class_names[c],
            "gt_count": counts[c]["gt"],
            "pred_count": counts[c]["pred"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    per_class_df = pd.DataFrame(rows)

    # Micro row
    total_tp = int(per_class_df["tp"].sum())
    total_fp = int(per_class_df["fp"].sum())
    total_fn = int(per_class_df["fn"].sum())

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0.0
    )

    micro_row = {
        "class_id": -1,
        "class_name": "micro_average",
        "gt_count": int(per_class_df["gt_count"].sum()),
        "pred_count": int(per_class_df["pred_count"].sum()),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1,
    }

    macro_row = {
        "class_id": -2,
        "class_name": "macro_average",
        "gt_count": int(per_class_df["gt_count"].sum()),
        "pred_count": int(per_class_df["pred_count"].sum()),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": float(per_class_df["precision"].mean()),
        "recall": float(per_class_df["recall"].mean()),
        "f1": float(per_class_df["f1"].mean()),
    }

    per_class_df = pd.concat(
        [per_class_df, pd.DataFrame([micro_row, macro_row])],
        ignore_index=True,
    )

    return per_class_df, pd.DataFrame(per_image_rows), pd.DataFrame(error_rows)


def compute_negative_image_metrics(per_image_df: pd.DataFrame) -> dict:
    neg = per_image_df[per_image_df["is_negative_image"] == True]
    pos = per_image_df[per_image_df["is_negative_image"] == False]

    num_negative = len(neg)
    num_positive = len(pos)

    neg_with_pred = int((neg["num_pred"] > 0).sum()) if num_negative > 0 else 0
    pos_with_pred = int((pos["num_pred"] > 0).sum()) if num_positive > 0 else 0

    return {
        "num_images": int(len(per_image_df)),
        "num_positive_images": int(num_positive),
        "num_negative_images": int(num_negative),

        "negative_images_with_any_prediction": neg_with_pred,
        "negative_image_false_positive_rate": (
            neg_with_pred / num_negative if num_negative > 0 else np.nan
        ),
        "negative_image_specificity": (
            1.0 - neg_with_pred / num_negative if num_negative > 0 else np.nan
        ),
        "false_positives_per_negative_image": (
            float(neg["fp"].sum()) / num_negative if num_negative > 0 else np.nan
        ),

        "positive_images_with_any_prediction": pos_with_pred,
        "positive_image_detection_rate": (
            pos_with_pred / num_positive if num_positive > 0 else np.nan
        ),
    }


# ---------------------------------------------------------------------
# Confusion matrix with background
# ---------------------------------------------------------------------

def compute_confusion_matrix_with_background(
    gt_by_image: Dict[str, dict],
    predictions_by_image: Dict[str, List[dict]],
    class_names: Dict[int, str],
    *,
    conf_threshold: float,
    iou_threshold: float,
) -> pd.DataFrame:
    """
    Class-agnostic matching first, then record GT class vs predicted class.

    Rows: actual class
    Cols: predicted class

    Includes background:
        actual background, predicted class = false positive
        actual class, predicted background = false negative
    """
    class_ids = sorted(class_names.keys())
    bg_id = max(class_ids) + 1

    matrix_ids = class_ids + [bg_id]
    id_to_name = {**class_names, bg_id: "background"}

    cm = pd.DataFrame(
        0,
        index=[id_to_name[i] for i in matrix_ids],
        columns=[id_to_name[i] for i in matrix_ids],
    )

    for image_id, gt_record in gt_by_image.items():
        gts = gt_record["gts"]
        preds = [
            p for p in predictions_by_image.get(image_id, [])
            if p["conf"] >= conf_threshold
        ]

        matched, unmatched_preds, unmatched_gts = match_predictions_to_ground_truths(
            gts=gts,
            preds=preds,
            iou_threshold=iou_threshold,
            class_aware=False,
        )

        for m in matched:
            gt = gts[m["gt_idx"]]
            pred = preds[m["pred_idx"]]

            actual = id_to_name[gt["class_id"]]
            predicted = id_to_name[pred["class_id"]]
            cm.loc[actual, predicted] += 1

        for pred in unmatched_preds:
            actual = "background"
            predicted = id_to_name[pred["class_id"]]
            cm.loc[actual, predicted] += 1

        for gt in unmatched_gts:
            actual = id_to_name[gt["class_id"]]
            predicted = "background"
            cm.loc[actual, predicted] += 1

    return cm


# ---------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------

def threshold_sweep(
    gt_by_image: Dict[str, dict],
    predictions_by_image: Dict[str, List[dict]],
    class_names: Dict[int, str],
    *,
    thresholds: List[float],
    iou_threshold: float,
) -> pd.DataFrame:
    rows = []

    for conf in tqdm(thresholds, desc="Threshold sweep"):
        per_class_df, per_image_df, _ = compute_object_metrics_at_threshold(
            gt_by_image,
            predictions_by_image,
            class_names,
            conf_threshold=conf,
            iou_threshold=iou_threshold,
        )

        micro = per_class_df[per_class_df["class_name"] == "micro_average"].iloc[0]
        macro = per_class_df[per_class_df["class_name"] == "macro_average"].iloc[0]
        neg_metrics = compute_negative_image_metrics(per_image_df)

        rows.append({
            "conf_threshold": conf,
            "iou_threshold": iou_threshold,

            "micro_precision": micro["precision"],
            "micro_recall": micro["recall"],
            "micro_f1": micro["f1"],

            "macro_precision": macro["precision"],
            "macro_recall": macro["recall"],
            "macro_f1": macro["f1"],

            "num_predictions": int(per_image_df["num_pred"].sum()),
            "num_gt": int(per_image_df["num_gt"].sum()),
            "total_tp": int(per_image_df["tp"].sum()),
            "total_fp": int(per_image_df["fp"].sum()),
            "total_fn": int(per_image_df["fn"].sum()),

            **neg_metrics,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

# python yolov8_evaluate.py \
# --model /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk/A_masked_2_full_orddc_weights_yolov8x_batch_16_lr0_0.001_lrf_0.01_imgsz_640_opt_SGD_lr1e-3/weights/best.pt \
# --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
# --split val \
# --output-dir /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk/A_masked_2_full_orddc_weights_yolov8x_batch_16_lr0_0.001_lrf_0.01_imgsz_640_opt_SGD_lr1e-3/evaluation/

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
        default=0.001,
        help="Low threshold for initial prediction collection."
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.7,
        help="NMS IoU used by Ultralytics prediction."
    )

    parser.add_argument(
        "--eval-conf",
        type=float,
        default=0.25,
        help="Main confidence threshold for detailed fixed-threshold reports."
    )
    parser.add_argument(
        "--eval-iou",
        type=float,
        default=0.50,
        help="IoU threshold for fixed-threshold detection matching."
    )

    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80",
        help="Comma-separated confidence thresholds for sweep."
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

    # Save raw predictions as JSON-friendly records
    raw_pred_rows = []
    for image_id, preds in predictions_by_image.items():
        for pred in preds:
            raw_pred_rows.append({
                "image_id": image_id,
                "class_id": pred["class_id"],
                "class_name": class_names.get(pred["class_id"], str(pred["class_id"])),
                "confidence": pred["conf"],
                "x1": pred["box"][0],
                "y1": pred["box"][1],
                "x2": pred["box"][2],
                "y2": pred["box"][3],
            })

    raw_predictions_df = pd.DataFrame(raw_pred_rows)
    raw_predictions_df.to_csv(output_dir / "raw_predictions.csv", index=False)

    # AP / mAP
    iou_thresholds = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
    ap_df = compute_map_table(
        gt_by_image,
        predictions_by_image,
        class_names,
        iou_thresholds=iou_thresholds,
    )
    ap_df.to_csv(output_dir / "ap_per_class_per_iou.csv", index=False)

    ap50_df = ap_df[ap_df["iou_threshold"] == 0.50]
    map50 = float(ap50_df["ap"].mean(skipna=True))

    map5095_by_class = (
        ap_df
        .groupby(["class_id", "class_name"], as_index=False)["ap"]
        .mean()
        .rename(columns={"ap": "ap50_95"})
    )
    map5095_by_class.to_csv(output_dir / "ap50_95_per_class.csv", index=False)

    map5095 = float(map5095_by_class["ap50_95"].mean(skipna=True))

    # Fixed threshold metrics
    per_class_df, per_image_df, errors_df = compute_object_metrics_at_threshold(
        gt_by_image,
        predictions_by_image,
        class_names,
        conf_threshold=args.eval_conf,
        iou_threshold=args.eval_iou,
    )

    per_class_df.to_csv(output_dir / "metrics_per_class_at_eval_threshold.csv", index=False)
    per_image_df.to_csv(output_dir / "metrics_per_image_at_eval_threshold.csv", index=False)
    errors_df.to_csv(output_dir / "errors_false_pos_false_neg.csv", index=False)

    neg_metrics = compute_negative_image_metrics(per_image_df)

    # Confusion matrix
    cm_df = compute_confusion_matrix_with_background(
        gt_by_image,
        predictions_by_image,
        class_names,
        conf_threshold=args.eval_conf,
        iou_threshold=args.eval_iou,
    )
    cm_df.to_csv(output_dir / "confusion_matrix_with_background.csv")

    # Threshold sweep
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    sweep_df = threshold_sweep(
        gt_by_image,
        predictions_by_image,
        class_names,
        thresholds=thresholds,
        iou_threshold=args.eval_iou,
    )
    sweep_df.to_csv(output_dir / "threshold_sweep.csv", index=False)

    best_micro_f1_row = sweep_df.iloc[sweep_df["micro_f1"].idxmax()].to_dict()
    best_macro_f1_row = sweep_df.iloc[sweep_df["macro_f1"].idxmax()].to_dict()

    summary = {
        "model": args.model,
        "data": args.data,
        "split": args.split,
        "num_images": len(image_paths),
        "num_positive_images": num_positive,
        "num_negative_images": num_negative,
        "num_gt_objects": num_gt_objects,
        "class_names": class_names,

        "prediction_conf_min": args.pred_conf_min,
        "prediction_nms_iou": args.nms_iou,
        "eval_conf": args.eval_conf,
        "eval_iou": args.eval_iou,

        "map50": map50,
        "map50_95": map5095,

        "negative_metrics_at_eval_threshold": neg_metrics,

        "best_micro_f1_threshold_row": best_micro_f1_row,
        "best_macro_f1_threshold_row": best_macro_f1_row,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()