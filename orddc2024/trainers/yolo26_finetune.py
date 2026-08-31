from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26 fine-tuning backend for ORDDC experiments."
    )

    parser.add_argument(
        "--option",
        type=str,
        default="A_full_orddc",
        help="Experiment/pretraining option identifier.",
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the dataset YAML.",
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help=(
            "Initial weights. May be an Ultralytics identifier such as "
            "yolo26x.pt or a path to a local checkpoint."
        ),
    )

    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Output project directory.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Exact run directory name supplied by the parent Trainer.",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)

    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="SGD")

    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--save-period", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")

    # Augmentation parameters shared with the YOLOv8 backend.
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)

    parser.add_argument("--degrees", type=float, default=60.0)
    parser.add_argument("--translate", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=10.0)
    parser.add_argument("--perspective", type=float, default=0.0005)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)

    parser.add_argument("--device", type=str, default="0")

    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional experiment tag retained in metadata.",
    )

    parser.add_argument(
        "--extra-args-json",
        type=str,
        default="{}",
        help=(
            "JSON object of additional keyword arguments forwarded directly "
            "to YOLO.train()."
        ),
    )

    return parser.parse_args()


def parse_extra_args(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "--extra-args-json must contain a valid JSON object."
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            "--extra-args-json must decode to a JSON object."
        )

    return parsed


def train(args: argparse.Namespace) -> None:
    print("=" * 72)
    print("YOLO26 training backend")
    print(f"Ultralytics weights/model: {args.weights}")
    print(f"Dataset:                  {args.data}")
    print(f"Project:                  {args.project}")
    print(f"Run name:                 {args.run_name}")
    print("=" * 72)

    model = YOLO(args.weights)

    # device = (
    #     args.device
    #     if torch.cuda.is_available()
    #     else "cpu"
    # )
    device = args.device

    extra_args = parse_extra_args(args.extra_args_json)

    reserved_keys = {
        "data",
        "task",
        "epochs",
        "imgsz",
        "device",
        "batch",
        "project",
        "name",
        "optimizer",
        "lr0",
        "lrf",
        "freeze",
        "patience",
        "save",
        "save_period",
        "exist_ok",
        "seed",
        "deterministic",
        "amp",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "fliplr",
        "flipud",
    }

    conflicts = sorted(
        reserved_keys.intersection(extra_args)
    )
    if conflicts:
        raise ValueError(
            "The following keys are controlled by TrainingConfig and cannot "
            f"be overridden through extra_args: {conflicts}"
        )

    train_kwargs: dict[str, Any] = {
        "data": args.data,
        "task": "detect",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "device": device,
        "batch": args.batch,
        "project": args.project,
        "name": args.run_name,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "cos_lr": True,
        "freeze": 0,
        "patience": args.patience,
        "save": True,
        "save_period": args.save_period,
        "exist_ok": args.exist_ok,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "amp": args.amp,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "fliplr": args.fliplr,
        "flipud": args.flipud,
    }

    train_kwargs.update(extra_args)

    start_time = time.time()
    model.train(**train_kwargs)
    elapsed_time = time.time() - start_time

    trainer_save_dir = getattr(
        getattr(model, "trainer", None),
        "save_dir",
        None,
    )

    run_dir = (
        Path(trainer_save_dir)
        if trainer_save_dir is not None
        else Path(args.project) / args.run_name
    )

    run_dir = run_dir.expanduser().resolve()

    weights_dir = run_dir / "weights"
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    best_weights_path = weights_dir / "best.pt"
    last_weights_path = weights_dir / "last.pt"
    results_file = run_dir / "results.csv"

    metadata = {
        "backend": "yolo26",
        "option": args.option,
        "run_name": args.run_name,
        "data": str(Path(args.data).expanduser().resolve()),
        "weights": args.weights,
        "project": str(Path(args.project).expanduser().resolve()),
        "run_dir": str(run_dir),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "cos_lr": True,
        "freeze": 0,
        "patience": args.patience,
        "save_period": args.save_period,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "amp": args.amp,
        "device_requested": args.device,
        "device_used": str(device),
        "tag": args.tag,

        "augmentation": {
            "hsv_h": args.hsv_h,
            "hsv_s": args.hsv_s,
            "hsv_v": args.hsv_v,
            "degrees": args.degrees,
            "translate": args.translate,
            "scale": args.scale,
            "shear": args.shear,
            "perspective": args.perspective,
            "fliplr": args.fliplr,
            "flipud": args.flipud,
        },

        "extra_args": extra_args,

        "elapsed_seconds": elapsed_time,
        "best_weights": str(best_weights_path),
        "last_weights": str(last_weights_path),
        "results_file": str(results_file),

        # The backend does not generate prediction caches itself. It only
        # establishes the standard artifact location that the control-side
        # TrainingResult/PredictionResult code will use later.
        "predictions_dir": str(predictions_dir),
    }

    metadata_path = run_dir / "training_metadata.json"
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    hours, remainder = divmod(
        elapsed_time,
        3600,
    )
    minutes, seconds = divmod(
        remainder,
        60,
    )

    print("#" * 72)
    print(
        f"Execution time: "
        f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    )
    print(f"Run directory:      {run_dir}")
    print(f"Best weights:       {best_weights_path}")
    print(f"Last weights:       {last_weights_path}")
    print(f"Results CSV:        {results_file}")
    print(f"Predictions dir:    {predictions_dir}")
    print(f"Training metadata:  {metadata_path}")
    print("#" * 72)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()