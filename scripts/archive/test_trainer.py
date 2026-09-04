from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from orddc2024.registry import get_trainer
from orddc2024.trainers.base_trainer import TrainingConfig, TrainingResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the process-backed YOLOv8 trainer."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/trainer_smoke_test"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="2",
        help="CUDA device such as 1, or 'cpu'.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--option", type=str, default="smoke_test")
    parser.add_argument(
        "--extra-args-json",
        type=str,
        default="{}",
    )
    return parser.parse_args()


def load_and_validate_dataset_yaml(path: Path) -> dict[str, Any]:
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
        raise ValueError(
            f"Dataset YAML is missing required key(s): {missing}"
        )

    names = data["names"]
    if not isinstance(names, (dict, list)):
        raise ValueError("'names' must be a mapping or list.")

    nc = int(data["nc"])
    if nc <= 0:
        raise ValueError("'nc' must be positive.")
    if len(names) != nc:
        raise ValueError(
            f"Dataset YAML declares nc={nc}, but names contains "
            f"{len(names)} entries."
        )

    return data


def parse_extra_args(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "--extra-args-json must contain valid JSON."
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            "--extra-args-json must decode to an object."
        )
    return parsed


def resolve_weights(value: str) -> str:
    expanded = Path(value).expanduser()

    if expanded.is_file():
        return str(expanded.resolve())

    if expanded.parent != Path("."):
        raise FileNotFoundError(
            f"Local weights not found: {expanded.resolve()}"
        )

    # Preserve bare Ultralytics identifiers like yolov8n.pt.
    return value


def verify_training_result(
    result: TrainingResult,
    requested_device: str,
) -> dict[str, Any]:
    failures: list[str] = []

    if result.backend != "yolov8":
        failures.append(
            f"backend={result.backend!r}, expected 'yolov8'"
        )

    if not result.run_dir.is_dir():
        failures.append(f"run_dir missing: {result.run_dir}")

    if result.best_weights is None or not result.best_weights.is_file():
        failures.append(f"best.pt missing: {result.best_weights}")

    if result.results_file is None or not result.results_file.is_file():
        failures.append(f"results.csv missing: {result.results_file}")

    if result.metadata_file is None or not result.metadata_file.is_file():
        failures.append(
            f"training_metadata.json missing: {result.metadata_file}"
        )

    if result.predictions_dir is None or not result.predictions_dir.is_dir():
        failures.append(
            f"predictions directory missing: {result.predictions_dir}"
        )

    metadata = result.metadata

    if metadata.get("backend") != "yolov8":
        failures.append(
            "training metadata backend is not 'yolov8'"
        )

    metadata_device = str(metadata.get("device_requested", ""))
    if metadata_device != str(requested_device):
        failures.append(
            "metadata device_requested does not match request: "
            f"{metadata_device!r} != {requested_device!r}"
        )

    if failures:
        raise AssertionError(
            "Trainer smoke-test verification failed:\n"
            + "\n".join(f"  - {failure}" for failure in failures)
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
        "results_file": str(result.results_file),
        "metadata_file": str(result.metadata_file),
        "predictions_dir": str(result.predictions_dir),
        "metrics": result.metrics,
        "metadata_device_requested": metadata_device,
        "metadata_device_used": str(metadata.get("device_used", "")),
    }


def main() -> int:
    args = parse_args()

    data_path = args.data.expanduser().resolve()
    project_path = args.project.expanduser().resolve()

    dataset_yaml = load_and_validate_dataset_yaml(data_path)
    weights = resolve_weights(args.weights)
    extra_args = parse_extra_args(args.extra_args_json)

    tag = (
        args.tag
        if args.tag
        else datetime.now().strftime("smoke_%Y%m%d_%H%M%S")
    )

    print("=" * 80)
    print("YOLOV8 TRAINER SMOKE TEST")
    print("=" * 80)
    print(f"Dataset YAML: {data_path}")
    print(f"Classes:      {dataset_yaml['nc']}")
    print(f"Weights:      {weights}")
    print(f"Project:      {project_path}")
    print(f"Device:       {args.device}")
    print(f"Epochs:       {args.epochs}")
    print(f"Image size:   {args.imgsz}")
    print(f"Batch:        {args.batch}")
    print(f"Tag:          {tag}")
    print("=" * 80)

    config = TrainingConfig(
        data=data_path,
        weights=weights,
        project=project_path,
        option=args.option,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        patience=args.patience,
        save_period=args.save_period,
        seed=args.seed,
        deterministic=args.deterministic,
        amp=args.amp,
        exist_ok=False,
        device=args.device,
        tag=tag,
        extra_args=extra_args,
    )

    trainer = get_trainer("yolov8")
    result = trainer.train(config)

    summary = verify_training_result(
        result,
        requested_device=args.device,
    )

    summary_path = result.run_dir / "trainer_smoke_test_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "dataset_yaml": str(data_path),
                "dataset_nc": int(dataset_yaml["nc"]),
                "initial_weights": weights,
                "requested_device": str(args.device),
                "training": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("YOLOV8 TRAINER SMOKE TEST PASSED")
    print("=" * 80)
    print(f"Run directory:   {result.run_dir}")
    print(f"Best weights:    {result.best_weights}")
    print(f"Results CSV:     {result.results_file}")
    print(f"Metadata:        {result.metadata_file}")
    print(f"Predictions dir: {result.predictions_dir}")
    print(f"Smoke summary:   {summary_path}")

    if result.metrics:
        print()
        print("Parsed training metrics:")
        for key, value in sorted(result.metrics.items()):
            print(f"  {key}: {value}")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())