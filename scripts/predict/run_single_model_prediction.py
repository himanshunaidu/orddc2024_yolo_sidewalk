from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from orddc2024.evaluation.custom_validator import CustomValidator
from orddc2024.registry import get_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run predictions for one model, save a PredictionResult .npz cache, "
            "and immediately evaluate it with CustomValidator."
        )
    )

    # Prediction inputs.
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)

    parser.add_argument("--backend", type=str, default="yolov8")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable FP16 inference explicitly. If omitted, backend default is used.",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--agnostic-nms",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    # CustomValidator options.
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help=(
            "Optional explicit GT label directory. If omitted, "
            "CustomValidator infers labels from the image paths."
        ),
    )
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for CustomValidator outputs. Default: "
            "<output-parent>/<output-stem>_validation."
        ),
    )
    parser.add_argument(
        "--validator-min-conf",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--validator-max-det",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--confusion-conf",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--confusion-iou",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--validator-device",
        type=str,
        default="cpu",
        help="Device used by CustomValidator matching, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-missing-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow missing GT files. Disabled by default so evaluation failures "
            "are not silently treated as empty labels."
        ),
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


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

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
    """
    Extract the same smoothed mean-over-classes F1 peak represented by
    Ultralytics' F1_curve.png, when the expected DetMetrics internals exist.
    """
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

    if (
        f1_curve.ndim != 2
        or f1_curve.shape[0] == 0
        or f1_curve.shape[1] != px.size
        or px.size == 0
    ):
        return None

    mean_f1 = f1_curve.mean(axis=0)

    try:
        from ultralytics.utils.metrics import smooth

        plotted_f1 = np.asarray(
            smooth(mean_f1, 0.05),
            dtype=np.float64,
        )
    except Exception:
        plotted_f1 = mean_f1

    best_index = int(np.argmax(plotted_f1))

    return {
        "f1": float(plotted_f1[best_index]),
        "confidence": float(px[best_index]),
    }


def main() -> int:
    args = parse_args()

    images_path = args.images.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    data_path = args.data.expanduser().resolve()

    labels_dir = (
        args.labels_dir.expanduser().resolve()
        if args.labels_dir is not None
        else None
    )

    eval_output_dir = (
        args.eval_output_dir.expanduser().resolve()
        if args.eval_output_dir is not None
        else output_path.parent / f"{output_path.stem}_validation"
    )

    if not images_path.exists():
        raise FileNotFoundError(
            f"Images path does not exist: {images_path}"
        )

    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be in [0, 1].")

    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be in [0, 1].")

    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive.")

    if args.batch <= 0:
        raise ValueError("--batch must be positive.")

    if args.max_det <= 0:
        raise ValueError("--max-det must be positive.")

    if not 0.0 <= args.validator_min_conf <= 1.0:
        raise ValueError("--validator-min-conf must be in [0, 1].")

    if args.validator_max_det <= 0:
        raise ValueError("--validator-max-det must be positive.")

    dataset = load_dataset_yaml(data_path)

    model_param = {
        "weight": args.weight,
        "img_size": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "augment": args.augment,
        "agnostic_nms": args.agnostic_nms,
        "batch": args.batch,
        "max_det": args.max_det,
    }

    if args.device is not None:
        model_param["device"] = args.device

    if args.half is not None:
        model_param["half"] = args.half

    print("=" * 80)
    print("SINGLE-MODEL PREDICTION + CUSTOM VALIDATION")
    print("=" * 80)

    for key, value in model_param.items():
        print(f"{key:18s}: {value}")

    print(f"{'images':18s}: {images_path}")
    print(f"{'data':18s}: {data_path}")
    print(f"{'prediction cache':18s}: {output_path}")
    print(f"{'validation output':18s}: {eval_output_dir}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    predictor = get_predictor(args.backend)

    predictor.load(
        [model_param],
        str(images_path),
    )

    predictions = predictor.predict()

    if len(predictions) != 1:
        raise RuntimeError(
            f"Expected exactly one PredictionResult, received {len(predictions)}."
        )

    result = predictions[0]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.save_npz(output_path)

    print()
    print("-" * 80)
    print("PREDICTION COMPLETE")
    print("-" * 80)
    print(f"Images:       {len(result.images)}")
    print(f"Detections:   {result.num_detections}")
    print(f"Saved cache:  {output_path}")

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------
    ground_truths = CustomValidator.load_ground_truths(
        result.images,
        labels_dir=labels_dir,
        allow_missing_files=args.allow_missing_ground_truth,
    )

    gt_instance_count = sum(
        len(record["cls"])
        for record in ground_truths.values()
    )

    # ------------------------------------------------------------------
    # Custom validation
    # ------------------------------------------------------------------
    eval_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    validator = CustomValidator(
        names=dataset["names"],
        save_dir=eval_output_dir,
        plots=args.plots,
        min_conf=args.validator_min_conf,
        max_det=args.validator_max_det,
        confusion_conf=args.confusion_conf,
        confusion_iou=args.confusion_iou,
        device=args.validator_device,
    )

    validation_result = validator.evaluate(
        result,
        ground_truths,
        ground_truth_box_format="xywhn",
        ground_truth_label_offset=0,
    )

    overall = dict(
        validation_result.overall
    )

    f1_peak = extract_smoothed_f1_peak(
        validator
    )

    summary = {
        "prediction": {
            "backend": args.backend,
            "weight": args.weight,
            "images": str(images_path),
            "model_params": model_param,
            "cache": str(output_path),
            "image_count": len(result.images),
            "detection_count": int(result.num_detections),
        },
        "ground_truth": {
            "dataset_yaml": str(data_path),
            "labels_dir": (
                str(labels_dir)
                if labels_dir is not None
                else None
            ),
            "instance_count": int(gt_instance_count),
        },
        "validator": {
            "min_conf": float(args.validator_min_conf),
            "max_det": int(args.validator_max_det),
            "confusion_conf": float(args.confusion_conf),
            "confusion_iou": float(args.confusion_iou),
            "device": args.validator_device,
            "plots": bool(args.plots),
        },
        "overall": json_safe(overall),
        "smoothed_f1_curve_peak": f1_peak,
        "per_class": json_safe(
            getattr(
                validation_result,
                "per_class",
                {},
            )
        ),
        "confusion_matrix": json_safe(
            validation_result.confusion_matrix
        ),
    }

    summary_path = (
        eval_output_dir
        / "validation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            json_safe(summary),
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("CUSTOM VALIDATION COMPLETE")
    print("=" * 80)
    print(f"GT instances:            {gt_instance_count}")

    for key in (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "fitness",
    ):
        if key in overall:
            print(
                f"{key:24s}: "
                f"{float(overall[key]):.6f}"
            )

    if f1_peak is not None:
        print(
            "F1 curve peak:           "
            f"{f1_peak['f1']:.6f} "
            f"at confidence "
            f"{f1_peak['confidence']:.6f}"
        )
    else:
        print(
            "F1 curve peak:           "
            "could not extract programmatically; "
            "inspect F1_curve.png"
        )

    print(f"Validation summary:       {summary_path}")
    print(f"Validation outputs:       {eval_output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
