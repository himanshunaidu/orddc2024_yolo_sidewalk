from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState


DEFAULT_METRIC_COLUMNS = (
    "metrics/mAP50-95(B)",
    "metrics/mAP_0.5:0.95(B)",
    "metrics/mAP_0.5:0.95",
    "metrics/mAP50-95",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run sequential Optuna hyperparameter tuning around "
            "yolov8_finetune.py and maximize validation mAP50-95."
        )
    )

    # Existing training script and fixed training inputs.
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("yolov8_finetune.py"),
        help="Path to the existing yolov8_finetune.py script.",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=None,
        help=(
            "Working directory used to launch the training script. Defaults "
            "to the training script's parent directory."
        ),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/optuna_yolo"),
        help="Directory in which the training script creates trial runs.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--save-period", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")

    # Full model-variant x pretraining-source grid.
    parser.add_argument(
        "--yolov8x-coco-weights",
        type=str,
        default="yolov8x.pt",
        help="COCO-pretrained YOLOv8x weights or model alias.",
    )
    parser.add_argument(
        "--yolov8x-orddc-weights",
        type=str,
        required=True,
        help="ORDDC-pretrained YOLOv8x checkpoint.",
    )
    # parser.add_argument(
    #     "--yolo26x-coco-weights",
    #     type=str,
    #     default="yolo26x.pt",
    #     help="COCO-pretrained YOLO26x weights or model alias.",
    # )
    # parser.add_argument(
    #     "--yolo26x-orddc-weights",
    #     type=str,
    #     required=True,
    #     help="ORDDC-pretrained YOLO26x checkpoint.",
    # )

    # Optuna study configuration.
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", type=str, default="sidewalk_yolo_hpo")
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///sidewalk_yolo_hpo.db",
        help="Optuna storage URL, e.g. sqlite:///sidewalk_yolo_hpo.db.",
    )
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument(
        "--metric-column",
        type=str,
        default=None,
        help=(
            "Exact results.csv column to maximize. If omitted, common "
            "Ultralytics mAP50-95 column names are tried."
        ),
    )

    # Training hyperparameter ranges.
    parser.add_argument("--lr0-min", type=float, default=1e-5)
    parser.add_argument("--lr0-max", type=float, default=1e-2)
    parser.add_argument("--lrf-min", type=float, default=1e-2)
    parser.add_argument("--lrf-max", type=float, default=1.0)

    # Augmentation ranges exposed by yolov8_finetune.py.
    parser.add_argument("--hsv-h-max", type=float, default=0.05)
    parser.add_argument("--hsv-s-max", type=float, default=0.90)
    parser.add_argument("--hsv-v-max", type=float, default=0.90)
    parser.add_argument("--degrees-max", type=float, default=30.0)
    parser.add_argument("--translate-max", type=float, default=0.50)
    parser.add_argument("--scale-min", type=float, default=0.10)
    parser.add_argument("--scale-max", type=float, default=0.90)
    parser.add_argument("--shear-max", type=float, default=10.0)
    parser.add_argument("--perspective-max", type=float, default=0.001)
    parser.add_argument("--fliplr-max", type=float, default=0.50)

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.train_script = args.train_script.expanduser().resolve()
    if not args.train_script.is_file():
        raise FileNotFoundError(f"Training script not found: {args.train_script}")

    args.data = args.data.expanduser().resolve()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    if args.working_dir is None:
        args.working_dir = args.train_script.parent
    else:
        args.working_dir = args.working_dir.expanduser().resolve()
    if not args.working_dir.is_dir():
        raise NotADirectoryError(f"Working directory not found: {args.working_dir}")

    args.project = args.project.expanduser().resolve()
    args.project.mkdir(parents=True, exist_ok=True)

    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch == 0:
        raise ValueError("--batch cannot be zero")

    positive_log_ranges = (
        ("lr0", args.lr0_min, args.lr0_max),
        ("lrf", args.lrf_min, args.lrf_max),
    )
    for name, low, high in positive_log_ranges:
        if not (0 < low < high):
            raise ValueError(f"Invalid {name} range: expected 0 < min < max")

    linear_ranges = (
        ("scale", args.scale_min, args.scale_max),
    )
    for name, low, high in linear_ranges:
        if low > high:
            raise ValueError(f"Invalid {name} range: min must be <= max")

    for value_name in (
        "hsv_h_max",
        "hsv_s_max",
        "hsv_v_max",
        "degrees_max",
        "translate_max",
        "shear_max",
        "perspective_max",
        "fliplr_max",
    ):
        if getattr(args, value_name) < 0:
            raise ValueError(f"--{value_name.replace('_', '-')} cannot be negative")

    # Check explicitly supplied local checkpoints. Plain aliases such as
    # yolov8x.pt may be downloaded/resolved by Ultralytics, so they are allowed.
    for name in ("yolov8x_orddc_weights",):
        value = getattr(args, name)
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found for --{name.replace('_', '-')}: {path}")
        setattr(args, name, str(path.resolve()))

    for name in ("yolov8x_coco_weights",):
        value = getattr(args, name)
        path = Path(value).expanduser()
        if path.is_file():
            setattr(args, name, str(path.resolve()))


def sanitize_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned or "study"


def build_weight_grid(args: argparse.Namespace) -> dict[tuple[str, str], str]:
    return {
        ("yolov8x", "coco"): args.yolov8x_coco_weights,
        ("yolov8x", "orddc"): args.yolov8x_orddc_weights,
        # ("yolo26x", "coco"): args.yolo26x_coco_weights,
        # ("yolo26x", "orddc"): args.yolo26x_orddc_weights,
    }


def sample_parameters(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_variant": trial.suggest_categorical(
            "model_variant", ["yolov8x"]#, "yolo26x"]
        ),
        "pretraining": trial.suggest_categorical(
            "pretraining", ["coco", "orddc"]
        ),
        "lr0": trial.suggest_float(
            "lr0", args.lr0_min, args.lr0_max, log=True
        ),
        "lrf": trial.suggest_float(
            "lrf", args.lrf_min, args.lrf_max, log=True
        ),
        "hsv_h": trial.suggest_float("hsv_h", 0.0, args.hsv_h_max),
        "hsv_s": trial.suggest_float("hsv_s", 0.0, args.hsv_s_max),
        "hsv_v": trial.suggest_float("hsv_v", 0.0, args.hsv_v_max),
        "degrees": trial.suggest_float("degrees", 0.0, args.degrees_max),
        "translate": trial.suggest_float(
            "translate", 0.0, args.translate_max
        ),
        "scale": trial.suggest_float(
            "scale", args.scale_min, args.scale_max
        ),
        "shear": trial.suggest_float("shear", 0.0, args.shear_max),
        "perspective": trial.suggest_float(
            "perspective", 0.0, args.perspective_max
        ),
        "fliplr": trial.suggest_float("fliplr", 0.0, args.fliplr_max),
    }


def build_training_command(
    args: argparse.Namespace,
    params: dict[str, Any],
    weights: str,
    tag: str,
) -> list[str]:
    option = f"optuna_{params['model_variant']}_{params['pretraining']}"

    command = [
        sys.executable,
        str(args.train_script),
        "--option",
        option,
        "--data",
        str(args.data),
        "--weights",
        weights,
        "--project",
        str(args.project),
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--lr0",
        repr(params["lr0"]),
        "--lrf",
        repr(params["lrf"]),
        "--optimizer",
        args.optimizer,
        "--patience",
        str(args.patience),
        "--save-period",
        str(args.save_period),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--freeze",
        "0",
        "--tag",
        tag,
        "--hsv-h",
        repr(params["hsv_h"]),
        "--hsv-s",
        repr(params["hsv_s"]),
        "--hsv-v",
        repr(params["hsv_v"]),
        "--degrees",
        repr(params["degrees"]),
        "--translate",
        repr(params["translate"]),
        "--scale",
        repr(params["scale"]),
        "--shear",
        repr(params["shear"]),
        "--perspective",
        repr(params["perspective"]),
        "--fliplr",
        repr(params["fliplr"]),
        # Vertical flips are deliberately fixed because upside-down sidewalk
        # images are not representative of the deployment domain.
        "--flipud",
        "0.0",
    ]

    if args.amp:
        command.append("--amp")
    if args.deterministic:
        command.append("--deterministic")

    return command


def find_trial_run_dir(project: Path, tag: str) -> Path:
    candidates = [
        path
        for path in project.glob(f"*_{tag}")
        if path.is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a training run in {project} ending with _{tag}"
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_objective_from_results(
    results_csv: Path,
    requested_column: str | None,
) -> tuple[float, str, int]:
    if not results_csv.is_file():
        raise FileNotFoundError(f"Ultralytics results.csv not found: {results_csv}")

    with results_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = [
            {
                str(key).strip(): value.strip() if isinstance(value, str) else value
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        ]

    if not rows:
        raise ValueError(f"No epoch rows found in {results_csv}")

    available_columns = set().union(*(row.keys() for row in rows))
    candidates = (
        (requested_column,)
        if requested_column is not None
        else DEFAULT_METRIC_COLUMNS
    )
    metric_column = next(
        (column for column in candidates if column in available_columns),
        None,
    )
    if metric_column is None:
        raise KeyError(
            "Could not find a validation mAP50-95 column in results.csv. "
            f"Available columns: {sorted(available_columns)}. "
            "Pass --metric-column with the exact column name."
        )

    best_value = -math.inf
    best_row_index = -1
    for row_index, row in enumerate(rows):
        raw_value = row.get(metric_column, "")
        if raw_value in {None, ""}:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > best_value:
            best_value = value
            best_row_index = row_index

    if best_row_index < 0:
        raise ValueError(
            f"Column {metric_column!r} contains no finite numeric values in {results_csv}"
        )

    return best_value, metric_column, best_row_index


def create_objective(
    args: argparse.Namespace,
    weight_grid: dict[tuple[str, str], str],
):
    study_tag = sanitize_tag(args.study_name)

    def objective(trial: optuna.Trial) -> float:
        params = sample_parameters(trial, args)
        weights = weight_grid[
            (params["model_variant"], params["pretraining"])
        ]
        tag = f"optuna_{study_tag}_trial_{trial.number:05d}"

        command = build_training_command(args, params, weights, tag)

        print("\n" + "=" * 88)
        print(f"Starting Optuna trial {trial.number}")
        print(f"Model variant: {params['model_variant']}")
        print(f"Pretraining: {params['pretraining']}")
        print(f"Weights: {weights}")
        print("Command:")
        print(" ".join(command))
        print("=" * 88 + "\n")

        subprocess.run(
            command,
            cwd=args.working_dir,
            check=True,
            env=os.environ.copy(),
        )

        run_dir = find_trial_run_dir(args.project, tag)
        results_csv = run_dir / "results.csv"
        objective_value, metric_column, best_row_index = read_objective_from_results(
            results_csv,
            args.metric_column,
        )

        trial.set_user_attr("weights", weights)
        trial.set_user_attr("run_dir", str(run_dir))
        trial.set_user_attr("results_csv", str(results_csv))
        trial.set_user_attr("best_weights", str(run_dir / "weights" / "best.pt"))
        trial.set_user_attr("metric_column", metric_column)
        trial.set_user_attr("best_results_row", best_row_index)

        print(
            f"Trial {trial.number} objective: {objective_value:.6f} "
            f"from {metric_column}"
        )
        return objective_value

    return objective


def export_study(study: optuna.Study, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_csv = output_dir / "optuna_trials.csv"
    with trials_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "number",
                "state",
                "value",
                "params",
                "user_attrs",
                "datetime_start",
                "datetime_complete",
                "duration_seconds",
            ],
        )
        writer.writeheader()
        for trial in study.trials:
            duration_seconds = (
                trial.duration.total_seconds()
                if trial.duration is not None
                else None
            )
            writer.writerow(
                {
                    "number": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    "params": json.dumps(trial.params, sort_keys=True),
                    "user_attrs": json.dumps(trial.user_attrs, sort_keys=True),
                    "datetime_start": trial.datetime_start,
                    "datetime_complete": trial.datetime_complete,
                    "duration_seconds": duration_seconds,
                }
            )

    completed_trials = [
        trial for trial in study.trials if trial.state == TrialState.COMPLETE
    ]
    summary: dict[str, Any] = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "num_trials": len(study.trials),
        "num_completed": len(completed_trials),
        "trials_csv": str(trials_csv),
    }

    if completed_trials:
        best = study.best_trial
        summary["best_trial"] = {
            "number": best.number,
            "value": best.value,
            "params": best.params,
            "user_attrs": best.user_attrs,
        }

    summary_path = output_dir / "optuna_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nStudy trials written to: {trials_csv}")
    print(f"Study summary written to: {summary_path}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    weight_grid = build_weight_grid(args)

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    objective = create_objective(args, weight_grid)
    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
        catch=(
            subprocess.CalledProcessError,
            FileNotFoundError,
            KeyError,
            ValueError,
            RuntimeError,
        ),
    )

    export_study(study, args.project / "study_exports")

    completed_trials = [
        trial for trial in study.trials if trial.state == TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError("The study completed without any successful trials")

    print("\nBest trial")
    print(f"  Number: {study.best_trial.number}")
    print(f"  Objective: {study.best_value:.6f}")
    print("  Parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    if study.best_trial.user_attrs:
        print("  Artifacts:")
        for key, value in study.best_trial.user_attrs.items():
            print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
    
# python optuna_yolo_tune.py \
#     --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
#     --yolov8x-coco-weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/models/models_coco/yolov8x_coco.pt \
#     --yolov8x-orddc-weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/orddc/orddc_175_16_960_weights_v8x_16_100_batch_16_lr0_0.001_lrf_0.01_imgsz_960_opt_SGD_lr1e-3/weights/best.pt
