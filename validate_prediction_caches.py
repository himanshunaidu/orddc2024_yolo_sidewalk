from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from orddc2024.evaluation.custom_validator import CustomValidator
from orddc2024.predictions.prediction_result import PredictionResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cached PredictionResult .npz files with CustomValidator."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="One or more PredictionResult .npz files.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Dataset YAML containing names and nc.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Optional explicit YOLO labels directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/prediction_validation"),
    )
    parser.add_argument("--min-conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--confusion-conf", type=float, default=0.25)
    parser.add_argument("--confusion-iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-missing-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ground-truth-label-offset",
        type=int,
        default=0,
    )
    return parser.parse_args()


def load_dataset_yaml(path: Path) -> tuple[dict[int, str], dict[str, Any]]:
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Dataset YAML must contain a top-level mapping.")

    if "names" not in data or "nc" not in data:
        raise ValueError("Dataset YAML must contain both `names` and `nc`.")

    names_value = data["names"]

    if isinstance(names_value, dict):
        names = {
            int(class_id): str(name)
            for class_id, name in names_value.items()
        }
    elif isinstance(names_value, list):
        names = {
            class_id: str(name)
            for class_id, name in enumerate(names_value)
        }
    else:
        raise ValueError("`names` must be a mapping or list.")

    nc = int(data["nc"])

    if len(names) != nc:
        raise ValueError(
            f"Dataset YAML declares nc={nc}, but names contains {len(names)} classes."
        )

    if sorted(names) != list(range(nc)):
        raise ValueError(
            "Dataset class names must use consecutive zero-based IDs."
        )

    return names, data


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def evaluate_prediction_cache(
    *,
    prediction_path: Path,
    names: dict[int, str],
    labels_dir: Path | None,
    output_dir: Path,
    min_conf: float,
    max_det: int,
    confusion_conf: float,
    confusion_iou: float,
    device: str,
    plots: bool,
    allow_missing_labels: bool,
    ground_truth_label_offset: int,
) -> dict[str, Any]:
    prediction_path = prediction_path.expanduser().resolve()

    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"Prediction cache not found: {prediction_path}"
        )

    print()
    print("=" * 80)
    print(f"EVALUATING: {prediction_path.name}")
    print("=" * 80)

    predictions = PredictionResult.load_npz(prediction_path)

    print(f"Images:     {predictions.num_images}")
    print(f"Detections: {predictions.num_detections}")

    ground_truths = CustomValidator.load_ground_truths(
        predictions.images,
        labels_dir=labels_dir,
        allow_missing_files=allow_missing_labels,
    )

    validation_dir = output_dir / prediction_path.stem

    validator = CustomValidator(
        names=names,
        save_dir=validation_dir,
        plots=plots,
        min_conf=min_conf,
        max_det=max_det,
        confusion_conf=confusion_conf,
        confusion_iou=confusion_iou,
        device=device,
    )

    result = validator.evaluate(
        predictions=predictions,
        ground_truths=ground_truths,
        ground_truth_box_format="xywhn",
        ground_truth_label_offset=ground_truth_label_offset,
    )

    overall = {
        str(key): float(value)
        for key, value in result.overall.items()
    }

    print()
    print("Overall metrics:")
    for key, value in overall.items():
        print(f"  {key}: {value:.6f}")

    result_json = {
        "prediction_cache": str(prediction_path),
        "num_images": predictions.num_images,
        "num_detections": predictions.num_detections,
        "prediction_metadata": make_json_safe(predictions.metadata),
        "evaluation": {
            "min_conf": min_conf,
            "max_det": max_det,
            "confusion_conf": confusion_conf,
            "confusion_iou": confusion_iou,
            "device": device,
            "ground_truth_label_offset": ground_truth_label_offset,
        },
        "overall": overall,
        "confusion_matrix": result.confusion_matrix.tolist(),
    }

    result_path = validation_dir / "validation_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result_json, indent=2),
        encoding="utf-8",
    )

    print(f"Saved result: {result_path}")

    return result_json


def write_comparison_csv(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    metric_names = sorted(
        {
            metric_name
            for result in results
            for metric_name in result["overall"]
        }
    )

    fieldnames = [
        "prediction_cache",
        "num_images",
        "num_detections",
        *metric_names,
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                "prediction_cache": result["prediction_cache"],
                "num_images": result["num_images"],
                "num_detections": result["num_detections"],
            }
            row.update(result["overall"])
            writer.writerow(row)


def main() -> int:
    args = parse_args()

    if not 0.0 <= args.min_conf <= 1.0:
        raise ValueError("--min-conf must be in [0, 1].")
    if args.max_det <= 0:
        raise ValueError("--max-det must be positive.")
    if not 0.0 <= args.confusion_conf <= 1.0:
        raise ValueError("--confusion-conf must be in [0, 1].")
    if not 0.0 <= args.confusion_iou <= 1.0:
        raise ValueError("--confusion-iou must be in [0, 1].")

    names, dataset_yaml = load_dataset_yaml(args.data)

    labels_dir = (
        args.labels_dir.expanduser().resolve()
        if args.labels_dir is not None
        else None
    )

    if labels_dir is not None and not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Labels directory not found: {labels_dir}"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PREDICTION CACHE VALIDATION")
    print("=" * 80)
    print(f"Dataset YAML: {args.data.expanduser().resolve()}")
    print(f"Classes:      {len(names)}")
    print(f"Caches:       {len(args.predictions)}")
    print(f"Labels dir:   {labels_dir if labels_dir else 'auto-infer'}")
    print(f"Output:       {output_dir}")
    print(f"Min conf:     {args.min_conf}")
    print(f"Max det:      {args.max_det}")
    print("=" * 80)

    results = []

    for prediction_path in args.predictions:
        results.append(
            evaluate_prediction_cache(
                prediction_path=prediction_path,
                names=names,
                labels_dir=labels_dir,
                output_dir=output_dir,
                min_conf=args.min_conf,
                max_det=args.max_det,
                confusion_conf=args.confusion_conf,
                confusion_iou=args.confusion_iou,
                device=args.device,
                plots=args.plots,
                allow_missing_labels=args.allow_missing_labels,
                ground_truth_label_offset=args.ground_truth_label_offset,
            )
        )

    comparison_json_path = output_dir / "validation_comparison.json"
    comparison_json_path.write_text(
        json.dumps(
            {
                "dataset_yaml": str(args.data.expanduser().resolve()),
                "dataset": make_json_safe(dataset_yaml),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    comparison_csv_path = output_dir / "validation_comparison.csv"
    write_comparison_csv(results, comparison_csv_path)

    print()
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Comparison JSON: {comparison_json_path}")
    print(f"Comparison CSV:  {comparison_csv_path}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())