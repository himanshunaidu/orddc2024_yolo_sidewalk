from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from orddc2024.registry import get_trainer
from orddc2024.trainers.base_trainer import TrainingConfig, TrainingResult


BACKEND = "yolo26"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially train every YOLO26 model listed in a model YAML "
            "on one dataset."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the ORDDC dataset YAML.",
    )

    parser.add_argument(
        "--models",
        type=Path,
        required=True,
        help="Model YAML containing models.yolo26 entries.",
    )

    parser.add_argument(
        "--project",
        type=Path,
        required=True,
        help="Directory under which all YOLO26 runs are saved.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device passed to TrainingConfig, e.g. 2, or 'cpu'.",
    )

    parser.add_argument(
        "--option",
        type=str,
        default="orddc_pretrain",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )

    return parser.parse_args()


def load_dataset_yaml(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Dataset YAML must contain a top-level mapping."
        )

    required = {
        "names",
        "nc",
        "path",
        "train",
        "val",
    }

    missing = sorted(
        required - set(data)
    )

    if missing:
        raise ValueError(
            f"Dataset YAML missing required key(s): {missing}"
        )

    names = data["names"]
    nc = int(data["nc"])

    if not isinstance(names, (dict, list)):
        raise ValueError(
            "'names' must be a mapping or list."
        )

    if nc <= 0:
        raise ValueError(
            "'nc' must be positive."
        )

    if len(names) != nc:
        raise ValueError(
            f"Dataset YAML declares nc={nc}, but names has "
            f"{len(names)} entries."
        )

    return data


def load_models_yaml(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Model YAML not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Model YAML must contain a top-level mapping."
        )

    models = data.get("models")

    if not isinstance(models, dict):
        raise ValueError(
            "Model YAML must contain a 'models' mapping."
        )

    entries = models.get(BACKEND)

    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"Model YAML must contain a non-empty "
            f"models.{BACKEND} list."
        )

    normalized: list[dict[str, Any]] = []

    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            normalized.append(
                {
                    "weight": entry,
                }
            )
            continue

        if not isinstance(entry, dict):
            raise ValueError(
                f"models.{BACKEND}[{index}] must be a string "
                "or mapping."
            )

        if "weight" not in entry:
            raise ValueError(
                f"models.{BACKEND}[{index}] is missing 'weight'."
            )

        normalized.append(
            dict(entry)
        )

    return normalized


def resolve_weight(
    value: str,
    *,
    yaml_dir: Path,
) -> str:
    path = Path(value).expanduser()

    if path.is_absolute():
        if not path.is_file():
            raise FileNotFoundError(
                f"Weight not found: {path}"
            )

        return str(path.resolve())

    if path.parent != Path("."):
        candidate = (
            yaml_dir / path
        ).resolve()

        if not candidate.is_file():
            raise FileNotFoundError(
                f"Weight not found: {candidate}"
            )

        return str(candidate)

    # Preserve bare Ultralytics identifiers such as yolo26n.pt.
    return value


def get_model_name(
    entry: dict[str, Any],
    index: int,
) -> str:
    if entry.get("name"):
        return str(entry["name"])

    stem = Path(
        str(entry["weight"])
    ).stem

    return stem or f"{BACKEND}_{index:02d}"


def validate_model_overrides(
    entry: dict[str, Any],
    *,
    model_name: str,
) -> None:
    if "epochs" in entry:
        if int(entry["epochs"]) <= 0:
            raise ValueError(
                f"{model_name}: epochs must be positive."
            )
    
    if "batch" in entry:
        if int(entry["batch"]) == 0:
            raise ValueError(
                f"{model_name}: batch cannot be 0."
            )

    if "lr0" in entry:
        if float(entry["lr0"]) <= 0:
            raise ValueError(
                f"{model_name}: lr0 must be positive."
            )

    if "lrf" in entry:
        if float(entry["lrf"]) <= 0:
            raise ValueError(
                f"{model_name}: lrf must be positive."
            )

    if "patience" in entry:
        if int(entry["patience"]) < 0:
            raise ValueError(
                f"{model_name}: patience cannot be negative."
            )

    if "optimizer" in entry:
        if not str(entry["optimizer"]).strip():
            raise ValueError(
                f"{model_name}: optimizer cannot be empty."
            )

    # Current TrainingConfig/YOLO26 backend does not expose freeze as a
    # configurable field; the backend already uses freeze=0.
    if "freeze" in entry:
        try:
            freeze = int(entry["freeze"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{model_name}: freeze must currently be 0."
            ) from exc

        if freeze != 0:
            raise ValueError(
                f"{model_name}: this runner currently supports only "
                "freeze: 0 because the YOLO26 backend hardcodes full "
                "fine-tuning."
            )


def build_training_config(
    *,
    entry: dict[str, Any],
    data_path: Path,
    project_path: Path,
    weights: str,
    option: str,
    imgsz: int,
    device: str,
) -> TrainingConfig:
    """
    Construct TrainingConfig using TrainingConfig defaults first, then
    explicitly override model-specific values when provided in the YAML.
    """

    # Base configuration. Any fields not mentioned here or below use the
    # defaults defined by TrainingConfig.
    kwargs: dict[str, Any] = {
        "data": data_path,
        "weights": weights,
        "project": project_path,
        "option": option,
        "imgsz": imgsz,
        "device": device,
    }

    # MODEL-SPECIFIC OVERRIDES FROM YAML
    if "epochs" in entry:
        kwargs["epochs"] = int(entry["epochs"])
    
    if "batch" in entry:
        kwargs["batch"] = int(entry["batch"])

    if "lr0" in entry:
        kwargs["lr0"] = float(entry["lr0"])

    if "lrf" in entry:
        kwargs["lrf"] = float(entry["lrf"])

    if "patience" in entry:
        kwargs["patience"] = int(entry["patience"])

    if "optimizer" in entry:
        kwargs["optimizer"] = str(entry["optimizer"])

    if "tag" in entry:
        kwargs["tag"] = str(entry["tag"])

    return TrainingConfig(
        **kwargs
    )


def get_effective_hyperparameters(
    config: TrainingConfig,
    entry: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epochs": config.epochs,
        "batch": config.batch,
        "lr0": config.lr0,
        "lrf": config.lrf,
        "patience": config.patience,
        "optimizer": config.optimizer,
        "freeze": int(entry.get("freeze", 0)),
        "tag": config.tag,
        "imgsz": config.imgsz,
        "device": str(config.device),
    }


def verify_result(
    result: TrainingResult,
) -> dict[str, Any]:
    failures: list[str] = []

    if result.backend != BACKEND:
        failures.append(
            f"backend={result.backend!r}, expected {BACKEND!r}"
        )

    if not result.run_dir.is_dir():
        failures.append(
            f"run_dir missing: {result.run_dir}"
        )

    if (
        result.best_weights is None
        or not result.best_weights.is_file()
    ):
        failures.append(
            f"best.pt missing: {result.best_weights}"
        )

    if (
        result.results_file is None
        or not result.results_file.is_file()
    ):
        failures.append(
            f"results.csv missing: {result.results_file}"
        )

    if (
        result.metadata_file is None
        or not result.metadata_file.is_file()
    ):
        failures.append(
            f"training_metadata.json missing: "
            f"{result.metadata_file}"
        )

    if failures:
        raise AssertionError(
            "Training result verification failed:\n"
            + "\n".join(
                f"  - {failure}"
                for failure in failures
            )
        )

    return {
        "backend": result.backend,
        "run_name": result.run_name,
        "run_dir": str(result.run_dir),
        "best_weights": str(result.best_weights),
        "last_weights": (
            str(result.last_weights)
            if result.last_weights is not None
            else None
        ),
        "results_file": (
            str(result.results_file)
            if result.results_file is not None
            else None
        ),
        "metadata_file": (
            str(result.metadata_file)
            if result.metadata_file is not None
            else None
        ),
        "metrics": result.metrics,
        "metadata": result.metadata,
    }


def write_summary(
    path: Path,
    *,
    data_path: Path,
    models_path: Path,
    project_path: Path,
    device: str,
    imgsz: int,
    option: str,
    models: list[dict[str, Any]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "backend": BACKEND,
                "dataset_yaml": str(data_path),
                "model_yaml": str(models_path),
                "project": str(project_path),
                "device": str(device),
                "imgsz": imgsz,
                "option": option,
                "models": models,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    data_path = args.data.expanduser().resolve()
    models_path = args.models.expanduser().resolve()
    project_path = args.project.expanduser().resolve()

    dataset = load_dataset_yaml(
        data_path
    )

    model_entries = load_models_yaml(
        models_path
    )

    project_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        project_path
        / "yolo26_orddc_pretraining_summary.json"
    )

    trainer = get_trainer(
        BACKEND
    )

    print("=" * 80)
    print("YOLO26 ORDDC PRETRAINING")
    print("=" * 80)
    print(f"Dataset:    {data_path}")
    print(f"Classes:    {dataset['nc']}")
    print(f"Model YAML: {models_path}")
    print(f"Models:     {len(model_entries)}")
    print(f"Project:    {project_path}")
    print(f"Device:     {args.device}")
    print(f"Image size: {args.imgsz}")
    print("=" * 80)

    summaries: list[dict[str, Any]] = []

    for index, entry in enumerate(
        model_entries
    ):
        name = get_model_name(
            entry,
            index,
        )

        validate_model_overrides(
            entry,
            model_name=name,
        )

        weights = resolve_weight(
            str(entry["weight"]),
            yaml_dir=models_path.parent,
        )

        config = build_training_config(
            entry=entry,
            data_path=data_path,
            project_path=project_path,
            weights=weights,
            option=args.option,
            imgsz=args.imgsz,
            device=args.device,
        )

        effective = get_effective_hyperparameters(
            config,
            entry,
        )

        print()
        print("#" * 80)
        print(
            f"[{index + 1}/{len(model_entries)}] {name}"
        )
        print(f"Weights:   {weights}")
        print(f"Epochs:    {effective['epochs']}")
        print(f"Batch:     {effective['batch']}")
        print(f"lr0:       {effective['lr0']}")
        print(f"lrf:       {effective['lrf']}")
        print(f"Patience:  {effective['patience']}")
        print(f"Optimizer: {effective['optimizer']}")
        print(f"Freeze:    {effective['freeze']}")
        print(f"Tag:       {effective['tag']}")
        print("#" * 80)

        item: dict[str, Any] = {
            "model_index": index,
            "model_name": name,
            "initial_weights": weights,
            "effective_hyperparameters": effective,
            "status": "running",
        }

        try:
            result = trainer.train(
                config
            )

            item.update(
                {
                    "status": "completed",
                    "training": verify_result(
                        result
                    ),
                }
            )

            print(f"Completed:    {name}")
            print(f"Best weights: {result.best_weights}")
            print(f"Run dir:      {result.run_dir}")

        except Exception as exc:
            item.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

            summaries.append(
                item
            )

            write_summary(
                summary_path,
                data_path=data_path,
                models_path=models_path,
                project_path=project_path,
                device=args.device,
                imgsz=args.imgsz,
                option=args.option,
                models=summaries,
            )

            print(
                f"FAILED: {name}: "
                f"{type(exc).__name__}: {exc}"
            )

            if not args.continue_on_error:
                raise

            continue

        summaries.append(
            item
        )

        write_summary(
            summary_path,
            data_path=data_path,
            models_path=models_path,
            project_path=project_path,
            device=args.device,
            imgsz=args.imgsz,
            option=args.option,
            models=summaries,
        )

    completed = sum(
        item["status"] == "completed"
        for item in summaries
    )

    failed = sum(
        item["status"] == "failed"
        for item in summaries
    )

    print()
    print("=" * 80)
    print("YOLO26 ORDDC PRETRAINING COMPLETE")
    print("=" * 80)
    print(f"Completed: {completed}")
    print(f"Failed:    {failed}")
    print(f"Summary:   {summary_path}")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )