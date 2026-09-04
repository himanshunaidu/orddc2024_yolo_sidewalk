from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ensemble_boxes import weighted_boxes_fusion

from orddc2024.predictions.prediction_result import PredictionResult
from orddc2024.registry import get_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cached multi-model prediction and Weighted Boxes Fusion."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Model YAML containing models.<backend> entries.",
    )

    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Validation image directory or image-list file.",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="yolov8",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/ensemble_predictions"),
    )

    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "WBF model weights in the same order as the YAML models. "
            "Default: equal weights."
        ),
    )

    parser.add_argument(
        "--wbf-iou",
        type=float,
        default=0.55,
        help="IoU threshold used by Weighted Boxes Fusion.",
    )

    parser.add_argument(
        "--skip-box-thr",
        type=float,
        default=0.001,
        help="WBF confidence threshold applied before fusion.",
    )

    parser.add_argument(
        "--post-conf",
        type=float,
        default=0.0,
        help="Optional confidence threshold applied after WBF.",
    )

    parser.add_argument(
        "--save-individual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save each model PredictionResult as its own .npz cache.",
    )
    
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag to append to the output directory.",
    )

    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config must contain a top-level mapping.")

    if not isinstance(config.get("models"), dict):
        raise ValueError("Config must contain a top-level 'models' mapping.")

    return config


def resolve_model_paths(
    model_params: list[dict[str, Any]],
    *,
    yaml_dir: Path,
) -> list[dict[str, Any]]:
    resolved = []

    for index, original in enumerate(model_params):
        if not isinstance(original, dict):
            raise ValueError(f"Model entry {index} must be a mapping.")

        if "weight" not in original:
            raise ValueError(f"Model entry {index} is missing 'weight'.")

        param = dict(original)
        value = str(param["weight"])
        path = Path(value).expanduser()

        if path.is_absolute():
            if not path.is_file():
                raise FileNotFoundError(f"Weight not found: {path}")
            param["weight"] = str(path.resolve())

        else:
            candidate = (yaml_dir / path).resolve()

            if candidate.is_file():
                param["weight"] = str(candidate)
            elif path.parent != Path("."):
                raise FileNotFoundError(
                    f"Relative weight not found: {candidate}"
                )
            else:
                # Preserve bare Ultralytics identifiers if intentionally used.
                param["weight"] = value

        resolved.append(param)

    return resolved


def model_cache_name(
    backend: str,
    model_index: int,
    model_param: dict[str, Any],
) -> str:
    stem = Path(str(model_param["weight"])).stem
    return f"{backend}_{model_index:02d}_{stem}.npz"


def validate_prediction_alignment(
    predictions: list[PredictionResult],
) -> list[str]:
    if not predictions:
        raise ValueError("No PredictionResult objects were produced.")

    images = predictions[0].images

    for index, result in enumerate(predictions[1:], start=1):
        if result.images != images:
            raise ValueError(
                f"PredictionResult {index} has a different image ordering."
            )

    return images


def fuse_predictions(
    predictions: list[PredictionResult],
    *,
    weights: list[float],
    iou_thr: float,
    skip_box_thr: float,
    post_conf: float,
) -> PredictionResult:
    images = validate_prediction_alignment(predictions)

    if len(weights) != len(predictions):
        raise ValueError(
            f"Got {len(weights)} WBF weights for "
            f"{len(predictions)} prediction sets."
        )

    if any(weight <= 0 for weight in weights):
        raise ValueError("Every WBF model weight must be positive.")

    ensemble_boxes: list[list[list[float]]] = []
    ensemble_scores: list[list[float]] = []
    ensemble_labels: list[list[int]] = []

    for image_index in range(len(images)):
        boxes_list = [
            result.boxes[image_index]
            for result in predictions
        ]
        scores_list = [
            result.scores[image_index]
            for result in predictions
        ]
        labels_list = [
            result.labels[image_index]
            for result in predictions
        ]

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )

        keep = scores >= post_conf

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        ensemble_boxes.append(
            boxes.astype(np.float32).tolist()
        )
        ensemble_scores.append(
            scores.astype(np.float32).tolist()
        )
        ensemble_labels.append(
            labels.astype(np.int64).tolist()
        )

    return PredictionResult(
        images=list(images),
        boxes=ensemble_boxes,
        scores=ensemble_scores,
        labels=ensemble_labels,
        metadata={
            "type": "weighted_boxes_fusion",
            "num_models": len(predictions),
            "weights": list(weights),
            "iou_thr": float(iou_thr),
            "skip_box_thr": float(skip_box_thr),
            "post_conf": float(post_conf),
            "members": [
                result.metadata
                for result in predictions
            ],
        },
    )


def main() -> int:
    args = parse_args()

    config_path = args.config.expanduser().resolve()
    images_path = args.images.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.tag:
        output_dir = output_dir / args.tag

    if not images_path.exists():
        raise FileNotFoundError(
            f"Images path does not exist: {images_path}"
        )

    if not 0.0 <= args.wbf_iou <= 1.0:
        raise ValueError("--wbf-iou must be in [0, 1].")

    if not 0.0 <= args.skip_box_thr <= 1.0:
        raise ValueError("--skip-box-thr must be in [0, 1].")

    if not 0.0 <= args.post_conf <= 1.0:
        raise ValueError("--post-conf must be in [0, 1].")

    config = load_config(config_path)
    raw_models = config["models"].get(args.backend, [])

    if not raw_models:
        raise ValueError(
            f"No models configured under models.{args.backend}"
        )

    if not isinstance(raw_models, list):
        raise ValueError(
            f"models.{args.backend} must be a list."
        )

    model_params = resolve_model_paths(
        raw_models,
        yaml_dir=config_path.parent,
    )

    wbf_weights = (
        list(args.weights)
        if args.weights is not None
        else [1.0] * len(model_params)
    )

    if len(wbf_weights) != len(model_params):
        raise ValueError(
            f"--weights supplied {len(wbf_weights)} values but the YAML "
            f"contains {len(model_params)} {args.backend} models."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MULTI-MODEL PREDICTION + WBF")
    print("=" * 80)
    print(f"Backend:       {args.backend}")
    print(f"Models:        {len(model_params)}")
    print(f"Images:        {images_path}")
    print(f"WBF weights:   {wbf_weights}")
    print(f"WBF IoU:       {args.wbf_iou}")
    print(f"Skip box thr:  {args.skip_box_thr}")
    print(f"Post conf:     {args.post_conf}")
    print("=" * 80)

    predictor = get_predictor(args.backend)
    predictor.load(
        model_params,
        str(images_path),
    )

    predictions = predictor.predict()

    if len(predictions) != len(model_params):
        raise RuntimeError(
            f"Expected {len(model_params)} PredictionResult objects, "
            f"received {len(predictions)}."
        )

    validate_prediction_alignment(predictions)

    individual_cache_paths = []

    if args.save_individual:
        for model_index, (model_param, result) in enumerate(
            zip(model_params, predictions)
        ):
            cache_path = (
                output_dir
                / model_cache_name(
                    args.backend,
                    model_index,
                    model_param,
                )
            )
            result.save_npz(cache_path)
            orddc_output_dir = (
                output_dir
                / Path(model_cache_name(
                    args.backend,
                    model_index,
                    model_param,
                )).stem
            )
            result.save_orddc_folder(output_dir=orddc_output_dir, dataset_root=images_path.parent)
            individual_cache_paths.append(str(cache_path))

            print(
                f"Saved model {model_index} cache: "
                f"{cache_path} "
                f"({result.num_detections} detections)"
            )

    ensemble = fuse_predictions(
        predictions,
        weights=wbf_weights,
        iou_thr=args.wbf_iou,
        skip_box_thr=args.skip_box_thr,
        post_conf=args.post_conf,
    )

    ensemble_path = output_dir / "ensemble_wbf.npz"
    ensemble.save_npz(ensemble_path)

    summary = {
        "backend": args.backend,
        "config": str(config_path),
        "images": str(images_path),
        "model_count": len(model_params),
        "model_weights": [
            str(param["weight"])
            for param in model_params
        ],
        "individual_prediction_caches": individual_cache_paths,
        "individual_detection_counts": [
            result.num_detections
            for result in predictions
        ],
        "ensemble_prediction_cache": str(ensemble_path),
        "ensemble_detection_count": ensemble.num_detections,
        "wbf": {
            "weights": wbf_weights,
            "iou_thr": args.wbf_iou,
            "skip_box_thr": args.skip_box_thr,
            "post_conf": args.post_conf,
        },
    }

    summary_path = output_dir / "ensemble_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("ENSEMBLE PREDICTION COMPLETE")
    print(f"Ensemble cache: {ensemble_path}")
    print(f"Summary:        {summary_path}")
    print(
        f"Ensemble detections: {ensemble.num_detections}"
    )
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())