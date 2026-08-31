from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from orddc2024.predictions.prediction_result import PredictionResult
from orddc2024.registry import (
    available_predictor_backends,
    get_predictor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test registered ORDDC predictors and cache their "
            "PredictionResult outputs."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/predictor_smoke_test"),
    )
    parser.add_argument(
        "--backends",
        nargs="*",
        default=None,
        help="Optional subset, e.g. --backends yolov8 yolo26.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for a fast smoke test.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Also test PredictionResult.save_orddc_folder().",
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Model YAML must contain a top-level mapping.")
    if not isinstance(config.get("models"), dict):
        raise ValueError("Model YAML must contain a top-level 'models' mapping.")
    return config


def resolve_model_params(
    model_params: list[dict[str, Any]],
    *,
    yaml_dir: Path,
) -> list[dict[str, Any]]:
    """Resolve local relative weight paths relative to the YAML file."""
    resolved = []

    for index, original in enumerate(model_params):
        if not isinstance(original, dict):
            raise ValueError(f"Model entry {index} must be a mapping.")
        if "weight" not in original:
            raise ValueError(f"Model entry {index} is missing 'weight'.")

        model_param = dict(original)
        weight_value = str(model_param["weight"])
        weight_path = Path(weight_value).expanduser()

        if weight_path.is_absolute():
            if not weight_path.is_file():
                raise FileNotFoundError(
                    f"Configured weight does not exist: {weight_path}"
                )
            model_param["weight"] = str(weight_path.resolve())
        else:
            candidate = (yaml_dir / weight_path).resolve()
            if candidate.is_file():
                model_param["weight"] = str(candidate)
            elif weight_path.parent != Path("."):
                raise FileNotFoundError(
                    f"Configured relative weight path does not exist: {candidate}"
                )
            else:
                # Allows bare Ultralytics identifiers such as yolov8x.pt.
                model_param["weight"] = weight_value

        resolved.append(model_param)

    return resolved


def make_cache_name(
    backend: str,
    model_param: dict[str, Any],
    model_index: int,
) -> str:
    weight_name = Path(str(model_param["weight"])).stem
    return f"{backend}_{model_index:02d}_{weight_name}"


def assert_round_trip(
    original: PredictionResult,
    loaded: PredictionResult,
) -> None:
    if original.images != loaded.images:
        raise AssertionError("NPZ round-trip changed image order/identities.")
    if original.labels != loaded.labels:
        raise AssertionError("NPZ round-trip changed labels.")
    if original.metadata != loaded.metadata:
        raise AssertionError("NPZ round-trip changed metadata.")

    for image_index in range(original.num_images):
        original_boxes = np.asarray(
            original.boxes[image_index], dtype=np.float32
        ).reshape(-1, 4)
        loaded_boxes = np.asarray(
            loaded.boxes[image_index], dtype=np.float32
        ).reshape(-1, 4)

        original_scores = np.asarray(
            original.scores[image_index], dtype=np.float32
        )
        loaded_scores = np.asarray(
            loaded.scores[image_index], dtype=np.float32
        )

        if not np.allclose(
            original_boxes, loaded_boxes, rtol=0.0, atol=1e-6
        ):
            raise AssertionError(
                f"NPZ round-trip changed boxes for image index {image_index}."
            )
        if not np.allclose(
            original_scores, loaded_scores, rtol=0.0, atol=1e-6
        ):
            raise AssertionError(
                f"NPZ round-trip changed scores for image index {image_index}."
            )


def summarize(result: PredictionResult) -> dict[str, Any]:
    counts = np.asarray(
        [len(image_boxes) for image_boxes in result.boxes],
        dtype=np.int64,
    )
    all_scores = np.asarray(
        [score for image_scores in result.scores for score in image_scores],
        dtype=np.float32,
    )

    return {
        "num_images": result.num_images,
        "num_detections": result.num_detections,
        "detections_per_image_mean": (
            float(counts.mean()) if len(counts) else 0.0
        ),
        "detections_per_image_min": (
            int(counts.min()) if len(counts) else 0
        ),
        "detections_per_image_max": (
            int(counts.max()) if len(counts) else 0
        ),
        "score_min": float(all_scores.min()) if len(all_scores) else None,
        "score_max": float(all_scores.max()) if len(all_scores) else None,
        "score_mean": float(all_scores.mean()) if len(all_scores) else None,
        "box_format": result.box_format,
        "label_offset": result.label_offset,
        "metadata": result.metadata,
    }


def test_backend(
    *,
    backend: str,
    model_params: list[dict[str, Any]],
    images_path: Path,
    output_dir: Path,
    max_images: int | None,
    save_txt: bool,
    dataset_root: Path | None,
) -> list[dict[str, Any]]:
    print()
    print("=" * 80)
    print(f"TESTING PREDICTOR BACKEND: {backend}")
    print(f"Models: {len(model_params)}")
    print("=" * 80)

    predictor = get_predictor(backend)
    predictor.load(model_params, str(images_path))

    if not predictor.images:
        raise RuntimeError(
            f"{backend}: Predictor.load() produced an empty image list."
        )

    if max_images is not None:
        if max_images <= 0:
            raise ValueError("--max-images must be positive.")
        predictor.images = predictor.images[:max_images]

    expected_images = list(predictor.images)

    print(f"Images to predict: {len(expected_images)}")
    results = predictor.predict()

    if len(results) != len(model_params):
        raise AssertionError(
            f"{backend}: expected {len(model_params)} PredictionResult "
            f"objects, got {len(results)}."
        )

    summaries = []

    for model_index, (model_param, result) in enumerate(
        zip(model_params, results)
    ):
        if not isinstance(result, PredictionResult):
            raise TypeError(
                f"{backend} model {model_index}: expected PredictionResult, "
                f"got {type(result).__name__}."
            )
        if result.images != expected_images:
            raise AssertionError(
                f"{backend} model {model_index}: image order mismatch."
            )

        name = make_cache_name(backend, model_param, model_index)
        cache_path = output_dir / f"{name}.npz"

        result.save_npz(cache_path)
        loaded = PredictionResult.load_npz(cache_path)
        assert_round_trip(result, loaded)

        if save_txt:
            assert dataset_root is not None
            text_dir = output_dir / f"{name}_txt"
            result.save_orddc_folder(
                text_dir,
                dataset_root=dataset_root,
                include_scores=True,
            )

        model_summary = summarize(result)
        model_summary.update(
            {
                "backend": backend,
                "model_index": model_index,
                "weight": str(model_param["weight"]),
                "cache": str(cache_path.resolve()),
                "npz_round_trip": "passed",
            }
        )
        summaries.append(model_summary)

        print(
            f"PASS model {model_index}: "
            f"images={model_summary['num_images']}, "
            f"detections={model_summary['num_detections']}, "
            f"mean/image={model_summary['detections_per_image_mean']:.2f}"
        )
        print(f"Cache: {cache_path}")

    return summaries


def main() -> int:
    args = parse_args()

    config_path = args.config.expanduser().resolve()
    images_path = args.images.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not images_path.exists():
        raise FileNotFoundError(f"Images path does not exist: {images_path}")

    if args.save_txt and args.dataset_root is None:
        raise ValueError("--dataset-root is required with --save-txt.")

    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root is not None
        else None
    )

    config = load_yaml_config(config_path)
    models_config = config["models"]

    registered = set(available_predictor_backends())
    requested_backends = (
        args.backends if args.backends else list(models_config.keys())
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Registered predictor backends:", sorted(registered))
    print("Requested predictor backends:", requested_backends)
    print("Output directory:", output_dir)

    all_summaries = []

    for backend in requested_backends:
        if backend not in registered:
            raise ValueError(
                f"Backend {backend!r} is not registered. "
                f"Available: {sorted(registered)}"
            )

        raw_model_params = models_config.get(backend, [])
        if not raw_model_params:
            print(f"Skipping {backend}: no configured models.")
            continue
        if not isinstance(raw_model_params, list):
            raise ValueError(f"models.{backend} must be a list.")

        model_params = resolve_model_params(
            raw_model_params,
            yaml_dir=config_path.parent,
        )

        all_summaries.extend(
            test_backend(
                backend=backend,
                model_params=model_params,
                images_path=images_path,
                output_dir=output_dir,
                max_images=args.max_images,
                save_txt=args.save_txt,
                dataset_root=dataset_root,
            )
        )

    if not all_summaries:
        raise RuntimeError("No predictor models were tested.")

    summary_path = output_dir / "predictor_smoke_test_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": str(config_path),
                "images": str(images_path),
                "models_tested": len(all_summaries),
                "results": all_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("ALL PREDICTOR TESTS PASSED")
    print(f"Models tested: {len(all_summaries)}")
    print(f"Summary: {summary_path}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())