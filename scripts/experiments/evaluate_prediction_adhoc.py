from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from ensemble_boxes import weighted_boxes_fusion
from torchvision.ops import nms

from orddc2024.evaluation.custom_validator import CustomValidator
from orddc2024.predictions.prediction_result import PredictionResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate raw cached detections against class-aware NMS, "
            "class-agnostic NMS, and single-model WBF."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--nms-iou", type=float, nargs="*", default=[0.40, 0.50, 0.60, 0.70])
    parser.add_argument("--nms-conf", type=float, nargs="*", default=[0.001])
    parser.add_argument("--nms-post-conf", type=float, nargs="*", default=[0.0])

    parser.add_argument("--agnostic-nms-iou", type=float, nargs="*", default=[0.40, 0.50, 0.60, 0.70])
    parser.add_argument(
        "--agnostic-nms-conf",
        type=float,
        nargs="*",
        default=[0.001],
    )
    parser.add_argument(
        "--agnostic-nms-post-conf",
        type=float,
        nargs="*",
        default=[0.0],
    )

    parser.add_argument("--wbf-iou", type=float, nargs="*", default=[0.40, 0.50, 0.55, 0.60, 0.70])
    parser.add_argument(
        "--wbf-skip-box-thr",
        type=float,
        nargs="*",
        default=[0.001],
    )
    parser.add_argument("--wbf-post-conf", type=float, nargs="*", default=[0.0])

    parser.add_argument("--validator-min-conf", type=float, default=0.001)
    parser.add_argument("--validator-max-det", type=int, default=3000)
    parser.add_argument("--confusion-conf", type=float, default=0.25)
    parser.add_argument("--confusion-iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def load_dataset_yaml(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Dataset YAML must contain a top-level mapping.")

    required = {"names", "nc", "path", "train", "val"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Dataset YAML is missing required key(s): {missing}")

    names = data["names"]
    if not isinstance(names, (dict, list)):
        raise ValueError("'names' must be a mapping or list.")

    nc = int(data["nc"])
    if nc <= 0:
        raise ValueError("'nc' must be positive.")
    if len(names) != nc:
        raise ValueError(
            f"Dataset YAML declares nc={nc}, but names has {len(names)} entries."
        )

    return data


def validate_probabilities(name: str, values: list[float]) -> None:
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} values must be in [0, 1], got {value}")


def fmt(value: float) -> str:
    return f"{value:.6g}".replace(".", "p").replace("-", "m")


def make_result(
    source: PredictionResult,
    *,
    boxes: list[list[list[float]]],
    scores: list[list[float]],
    labels: list[list[int]],
    postprocessing: dict[str, Any],
) -> PredictionResult:
    return PredictionResult(
        images=list(source.images),
        boxes=boxes,
        scores=scores,
        labels=labels,
        metadata={
            **source.metadata,
            "postprocessing": postprocessing,
        },
    )


def apply_nms(
    source: PredictionResult,
    *,
    iou_thr: float,
    pre_conf: float,
    post_conf: float,
    class_agnostic: bool,
) -> PredictionResult:
    out_boxes: list[list[list[float]]] = []
    out_scores: list[list[float]] = []
    out_labels: list[list[int]] = []

    for image_index, image_path in enumerate(source.images):
        boxes = np.asarray(source.boxes[image_index], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(source.scores[image_index], dtype=np.float32).reshape(-1)
        labels = np.asarray(source.labels[image_index], dtype=np.int64).reshape(-1)

        if not (len(boxes) == len(scores) == len(labels)):
            raise ValueError(f"Mismatched detections for {image_path}")

        keep_conf = scores >= pre_conf
        boxes = boxes[keep_conf]
        scores = scores[keep_conf]
        labels = labels[keep_conf]

        if len(scores) == 0:
            out_boxes.append([])
            out_scores.append([])
            out_labels.append([])
            continue

        order = np.argsort(-scores)
        boxes = boxes[order]
        scores = scores[order]
        labels = labels[order]

        box_tensor = torch.from_numpy(boxes)
        score_tensor = torch.from_numpy(scores)

        if class_agnostic:
            keep = nms(box_tensor, score_tensor, iou_thr).cpu().numpy()
        else:
            kept_parts: list[np.ndarray] = []
            for class_id in np.unique(labels):
                class_indices = np.flatnonzero(labels == class_id)
                class_keep = nms(
                    box_tensor[class_indices],
                    score_tensor[class_indices],
                    iou_thr,
                ).cpu().numpy()
                kept_parts.append(class_indices[class_keep])

            keep = (
                np.concatenate(kept_parts)
                if kept_parts
                else np.empty(0, dtype=np.int64)
            )
            if len(keep):
                keep = keep[np.argsort(-scores[keep])]

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        keep_post = scores >= post_conf
        boxes = boxes[keep_post]
        scores = scores[keep_post]
        labels = labels[keep_post]

        out_boxes.append(boxes.astype(np.float32).tolist())
        out_scores.append(scores.astype(np.float32).tolist())
        out_labels.append(labels.astype(np.int64).tolist())

    return make_result(
        source,
        boxes=out_boxes,
        scores=out_scores,
        labels=out_labels,
        postprocessing={
            "type": "agnostic_nms" if class_agnostic else "class_aware_nms",
            "iou_thr": float(iou_thr),
            "pre_conf": float(pre_conf),
            "post_conf": float(post_conf),
        },
    )


def apply_wbf(
    source: PredictionResult,
    *,
    iou_thr: float,
    skip_box_thr: float,
    post_conf: float,
) -> PredictionResult:
    out_boxes: list[list[list[float]]] = []
    out_scores: list[list[float]] = []
    out_labels: list[list[int]] = []

    for image_index in range(len(source.images)):
        boxes, scores, labels = weighted_boxes_fusion(
            [source.boxes[image_index]],
            [source.scores[image_index]],
            [source.labels[image_index]],
            weights=[1.0],
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )

        keep = scores >= post_conf
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        out_boxes.append(boxes.astype(np.float32).tolist())
        out_scores.append(scores.astype(np.float32).tolist())
        out_labels.append(labels.astype(np.int64).tolist())

    return make_result(
        source,
        boxes=out_boxes,
        scores=out_scores,
        labels=out_labels,
        postprocessing={
            "type": "single_model_wbf",
            "weights": [1.0],
            "iou_thr": float(iou_thr),
            "skip_box_thr": float(skip_box_thr),
            "post_conf": float(post_conf),
        },
    )


def f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return 0.0 if denom <= 0 else 2.0 * precision * recall / denom


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def evaluate_variant(
    *,
    name: str,
    predictions: PredictionResult,
    names: Any,
    ground_truths: Any,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)

    cache_path = variant_dir / "predictions.npz"
    predictions.save_npz(cache_path)

    validator = CustomValidator(
        names=names,
        save_dir=variant_dir,
        plots=args.plots,
        min_conf=args.validator_min_conf,
        max_det=args.validator_max_det,
        confusion_conf=args.confusion_conf,
        confusion_iou=args.confusion_iou,
        device=args.device,
    )

    result = validator.evaluate(
        predictions,
        ground_truths,
    )

    overall = dict(result.overall)
    precision = float(overall.get("metrics/precision(B)", 0.0))
    recall = float(overall.get("metrics/recall(B)", 0.0))

    row = {
        "variant": name,
        "detections": int(predictions.num_detections),
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
        "mAP50": float(overall.get("metrics/mAP50(B)", 0.0)),
        "mAP50-95": float(overall.get("metrics/mAP50-95(B)", 0.0)),
        "fitness": float(overall.get("fitness", 0.0)),
        "postprocessing": predictions.metadata.get("postprocessing", {}),
        "cache": str(cache_path),
        "per_class": json_safe(getattr(result, "per_class", {})),
        "confusion_matrix": json_safe(result.confusion_matrix),
    }

    (variant_dir / "validation_result.json").write_text(
        json.dumps(json_safe(row), indent=2),
        encoding="utf-8",
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "variant",
        "detections",
        "precision",
        "recall",
        "f1",
        "mAP50",
        "mAP50-95",
        "fitness",
        "cache",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def main() -> int:
    args = parse_args()

    predictions_path = args.predictions.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    labels_dir = (
        args.labels_dir.expanduser().resolve()
        if args.labels_dir is not None
        else None
    )

    if not predictions_path.is_file():
        raise FileNotFoundError(f"Prediction cache not found: {predictions_path}")

    dataset = load_dataset_yaml(data_path)
    source = PredictionResult.load_npz(predictions_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ground_truths = CustomValidator.load_ground_truths(
        source.images,
        labels_dir=labels_dir,
        allow_missing_files=True,
    )
    
    # print(len(source.images))
    print("Validation images:", len(source.images))
    print(
        "Validation GT instances:",
        sum(len(record["cls"]) for record in ground_truths.values())
    )


if __name__ == "__main__":
    raise SystemExit(main())
