from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from orddc2024.evaluation.custom_validator import CustomValidator
from orddc2024.predictions.prediction_result import PredictionResult


IMAGE_EXTENSIONS = {
    ".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png",
    ".tif", ".tiff", ".webp", ".pfm", ".heic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert native Ultralytics model.val() save_txt predictions into "
            "PredictionResult format and evaluate them with CustomValidator."
        )
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help=(
            "Ultralytics validation labels directory produced by "
            "model.val(save_txt=True, save_conf=True)."
        )
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Validation image-list file (e.g. val.txt) or image directory."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Dataset YAML used for validation; used for class names/path resolution."
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help=(
            "Optional explicit GT label directory. If omitted, "
            "CustomValidator.load_ground_truths() infers label paths."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for cache, plots, and summary JSON.",
    )
    parser.add_argument(
        "--cache-name",
        type=str,
        default="native_ultralytics_val_predictions.npz",
    )
    parser.add_argument("--validator-min-conf", type=float, default=0.001)
    parser.add_argument("--validator-max-det", type=int, default=300)
    parser.add_argument("--confusion-conf", type=float, default=0.25)
    parser.add_argument("--confusion-iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-missing-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disabled by default for strict parity testing.",
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
    if "names" not in data:
        raise ValueError("Dataset YAML is missing 'names'.")
    if not isinstance(data["names"], (dict, list)):
        raise ValueError("Dataset YAML 'names' must be a mapping or list.")
    return data


def resolve_dataset_root(data: dict[str, Any], data_yaml: Path) -> Path:
    raw_root = data.get("path")
    if raw_root is None:
        return data_yaml.parent.resolve()

    root = Path(str(raw_root)).expanduser()
    if root.is_absolute():
        return root.resolve()
    return (data_yaml.parent / root).resolve()


def resolve_list_image(
    raw_value: str,
    *,
    list_file: Path,
    dataset_root: Path,
) -> Path:
    value = raw_value.strip()
    if not value:
        raise ValueError("Encountered an empty image-list entry.")

    path = Path(value).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Image not found: {resolved}")
        return resolved

    cleaned = value[2:] if value.startswith("./") else value
    candidates = [
        list_file.parent / cleaned,
        dataset_root / cleaned,
        Path.cwd() / cleaned,
    ]

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve image-list entry {value!r}. Tried:\n"
        + "\n".join(f"  - {candidate.resolve()}" for candidate in candidates)
    )


def load_images(images_source: Path, dataset_root: Path) -> list[str]:
    images_source = images_source.expanduser().resolve()

    if images_source.is_dir():
        images = sorted(
            str(path.resolve())
            for path in images_source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    elif images_source.is_file():
        images = []
        for raw_line in images_source.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            images.append(
                str(
                    resolve_list_image(
                        line,
                        list_file=images_source,
                        dataset_root=dataset_root,
                    )
                )
            )
    else:
        raise FileNotFoundError(f"Images source not found: {images_source}")

    if not images:
        raise ValueError(f"No images found from: {images_source}")

    # Ultralytics save_txt() uses only the image stem for each prediction file.
    by_stem: dict[str, list[str]] = {}
    for image in images:
        by_stem.setdefault(Path(image).stem, []).append(image)

    duplicates = {stem: paths for stem, paths in by_stem.items() if len(paths) > 1}
    if duplicates:
        details = "\n".join(
            f"  {stem}: {paths}" for stem, paths in sorted(duplicates.items())
        )
        raise ValueError(
            "Duplicate image stems make Ultralytics save_txt prediction lookup "
            f"ambiguous:\n{details}"
        )

    return images


def xywhn_to_xyxyn(
    xc: float,
    yc: float,
    width: float,
    height: float,
) -> list[float]:
    return [
        xc - width / 2.0,
        yc - height / 2.0,
        xc + width / 2.0,
        yc + height / 2.0,
    ]


def load_ultralytics_val_predictions(
    images: list[str],
    predictions_dir: Path,
) -> PredictionResult:
    predictions_dir = predictions_dir.expanduser().resolve()
    if not predictions_dir.is_dir():
        raise NotADirectoryError(
            f"Ultralytics predictions directory not found: {predictions_dir}"
        )

    all_boxes: list[list[list[float]]] = []
    all_scores: list[list[float]] = []
    all_labels: list[list[int]] = []

    prediction_files_found = 0
    images_without_prediction_file = 0

    for image in images:
        prediction_file = predictions_dir / f"{Path(image).stem}.txt"

        image_boxes: list[list[float]] = []
        image_scores: list[float] = []
        image_labels: list[int] = []

        # Ultralytics may not create a txt file for an image with zero detections.
        if prediction_file.is_file():
            prediction_files_found += 1

            for line_number, raw_line in enumerate(
                prediction_file.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                line = raw_line.strip()
                if not line:
                    continue

                values = line.split()
                if len(values) != 6:
                    raise ValueError(
                        f"{prediction_file}:{line_number}: expected 6 columns "
                        "(class xc yc w h confidence), but got "
                        f"{len(values)}. Did you use save_conf=True?"
                    )

                try:
                    class_value = float(values[0])
                    xc, yc, width, height, confidence = map(float, values[1:])
                except ValueError as error:
                    raise ValueError(
                        f"{prediction_file}:{line_number}: invalid numeric data: {line!r}"
                    ) from error

                class_id = int(class_value)
                if class_value != class_id:
                    raise ValueError(
                        f"{prediction_file}:{line_number}: non-integer class ID "
                        f"{class_value}."
                    )
                if width < 0.0 or height < 0.0:
                    raise ValueError(
                        f"{prediction_file}:{line_number}: negative box size."
                    )
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError(
                        f"{prediction_file}:{line_number}: confidence {confidence} "
                        "is outside [0, 1]."
                    )

                image_boxes.append(xywhn_to_xyxyn(xc, yc, width, height))
                image_scores.append(confidence)
                image_labels.append(class_id)
        else:
            images_without_prediction_file += 1

        all_boxes.append(image_boxes)
        all_scores.append(image_scores)
        all_labels.append(image_labels)

    return PredictionResult(
        images=list(images),
        boxes=all_boxes,
        scores=all_scores,
        labels=all_labels,
        metadata={
            "backend": "ultralytics_native_val",
            "source_predictions_dir": str(predictions_dir),
            "source_format": "ultralytics_save_txt_with_conf",
            "source_box_format": "xywhn",
            "canonical_box_format": "xyxyn",
            "label_offset": 0,
            "prediction_files_found": prediction_files_found,
            "images_without_prediction_file": images_without_prediction_file,
        },
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def extract_smoothed_f1_peak(
    validator: CustomValidator,
) -> dict[str, float] | None:
    """Extract the same smoothed mean-F1 peak used by Ultralytics F1_curve.png."""
    metrics = getattr(validator, "metrics", None)
    box_metrics = getattr(metrics, "box", None)
    if box_metrics is None:
        return None

    px = getattr(box_metrics, "px", None)
    f1_curve = getattr(box_metrics, "f1_curve", None)
    if px is None or f1_curve is None:
        return None

    px = np.asarray(px, dtype=np.float64)
    f1_curve = np.asarray(f1_curve, dtype=np.float64)

    if f1_curve.ndim != 2 or f1_curve.shape[0] == 0:
        return None
    if px.size == 0 or f1_curve.shape[1] != px.size:
        return None

    mean_f1 = f1_curve.mean(axis=0)

    try:
        from ultralytics.utils.metrics import smooth
        plotted_curve = np.asarray(smooth(mean_f1, 0.05), dtype=np.float64)
    except Exception:
        plotted_curve = mean_f1

    best_index = int(np.argmax(plotted_curve))
    return {
        "f1": float(plotted_curve[best_index]),
        "confidence": float(px[best_index]),
    }


def main() -> int:
    args = parse_args()

    predictions_dir = args.predictions_dir.expanduser().resolve()
    images_source = args.images.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    labels_dir = (
        args.labels_dir.expanduser().resolve()
        if args.labels_dir is not None
        else None
    )

    if not 0.0 <= args.validator_min_conf <= 1.0:
        raise ValueError("--validator-min-conf must be in [0, 1].")
    if args.validator_max_det <= 0:
        raise ValueError("--validator-max-det must be positive.")
    if not 0.0 <= args.confusion_conf <= 1.0:
        raise ValueError("--confusion-conf must be in [0, 1].")
    if not 0.0 <= args.confusion_iou <= 1.0:
        raise ValueError("--confusion-iou must be in [0, 1].")

    dataset = load_dataset_yaml(data_path)
    dataset_root = resolve_dataset_root(dataset, data_path)
    images = load_images(images_source, dataset_root)

    predictions = load_ultralytics_val_predictions(images, predictions_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / args.cache_name
    predictions.save_npz(cache_path)

    ground_truths = CustomValidator.load_ground_truths(
        predictions.images,
        labels_dir=labels_dir,
        allow_missing_files=args.allow_missing_ground_truth,
    )

    gt_instance_count = sum(len(record["cls"]) for record in ground_truths.values())

    validator = CustomValidator(
        names=dataset["names"],
        save_dir=output_dir,
        plots=args.plots,
        min_conf=args.validator_min_conf,
        max_det=args.validator_max_det,
        confusion_conf=args.confusion_conf,
        confusion_iou=args.confusion_iou,
        device=args.device,
    )

    print("=" * 80)
    print("NATIVE ULTRALYTICS PREDICTIONS -> CUSTOM VALIDATOR")
    print("=" * 80)
    print(f"Dataset YAML:             {data_path}")
    print(f"Dataset root:             {dataset_root}")
    print(f"Images source:            {images_source}")
    print(f"Images:                   {len(predictions.images)}")
    print(f"Ground-truth instances:   {gt_instance_count}")
    print(f"Ultralytics labels dir:   {predictions_dir}")
    print(f"Prediction txt files:     {predictions.metadata['prediction_files_found']}")
    print(
        "Images with no txt file:  "
        f"{predictions.metadata['images_without_prediction_file']}"
    )
    print(f"Total detections:         {predictions.num_detections}")
    print(f"PredictionResult cache:   {cache_path}")
    print(f"Validation output:        {output_dir}")
    print("=" * 80)

    result = validator.evaluate(
        predictions,
        ground_truths,
        ground_truth_box_format="xywhn",
        ground_truth_label_offset=0,
    )

    overall = dict(result.overall)
    f1_peak = extract_smoothed_f1_peak(validator)

    summary = {
        "input": {
            "dataset_yaml": str(data_path),
            "dataset_root": str(dataset_root),
            "images_source": str(images_source),
            "predictions_dir": str(predictions_dir),
            "labels_dir": str(labels_dir) if labels_dir is not None else None,
        },
        "counts": {
            "images": len(predictions.images),
            "ground_truth_instances": int(gt_instance_count),
            "prediction_txt_files_found": int(
                predictions.metadata["prediction_files_found"]
            ),
            "images_without_prediction_file": int(
                predictions.metadata["images_without_prediction_file"]
            ),
            "detections": int(predictions.num_detections),
        },
        "validator": {
            "min_conf": float(args.validator_min_conf),
            "max_det": int(args.validator_max_det),
            "confusion_conf": float(args.confusion_conf),
            "confusion_iou": float(args.confusion_iou),
            "device": args.device,
            "plots": bool(args.plots),
        },
        "overall": json_safe(overall),
        "smoothed_f1_curve_peak": f1_peak,
        "per_class": json_safe(getattr(result, "per_class", {})),
        "confusion_matrix": json_safe(result.confusion_matrix),
        "prediction_cache": str(cache_path),
    }

    summary_path = output_dir / "native_val_custom_validation.json"
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("CUSTOM VALIDATION COMPLETE")
    print("=" * 80)

    for key in (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "fitness",
    ):
        if key in overall:
            print(f"{key:24s}: {float(overall[key]):.6f}")

    if f1_peak is not None:
        print(
            "F1 curve peak:           "
            f"{f1_peak['f1']:.6f} at confidence {f1_peak['confidence']:.6f}"
        )
    else:
        print(
            "F1 curve peak:           could not extract programmatically; "
            "inspect F1_curve.png"
        )

    print(f"Summary JSON:             {summary_path}")
    print(f"Prediction cache:         {cache_path}")
    if args.plots:
        print(f"Plots directory:          {output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())