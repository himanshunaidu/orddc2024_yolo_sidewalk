from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from config.backends import BACKENDS


@dataclass(slots=True)
class TrainingConfig:
    """
    Framework-independent training configuration.

    These fields cover the parameters currently used by yolov8_finetune.py.
    Model-specific trainers may ignore unsupported fields or use `extra_args`
    for backend-specific options.
    """

    data: str | Path
    weights: str | Path
    project: str | Path

    option: str = "A_full_orddc"

    epochs: int = 100
    imgsz: int = 640
    batch: int = 32

    lr0: float = 0.001
    lrf: float = 0.01
    optimizer: str = "SGD"

    patience: int = 25
    save_period: int = 25
    seed: int = 42

    deterministic: bool = False
    amp: bool = False
    exist_ok: bool = False

    # Augmentation parameters currently exposed by yolov8_finetune.py.
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4

    degrees: float = 60.0
    translate: float = 0.5
    scale: float = 0.5
    shear: float = 10.0
    perspective: float = 0.0005
    fliplr: float = 0.5
    flipud: float = 0.0

    device: str = "0"
    tag: str = ""

    # Escape hatch for future trainer-specific settings without immediately
    # changing the shared TrainingConfig interface.
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingResult:
    """
    Standard result returned by every Trainer implementation.

    `metrics` is intentionally generic so different training frameworks can
    expose the metrics they produce without changing the orchestration layer.
    """

    backend: str
    run_name: str
    run_dir: Path

    best_weights: Path | None = None
    last_weights: Path | None = None

    results_file: Path | None = None
    metadata_file: Path | None = None

    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class Trainer(ABC):
    """
    Base class for process-backed model trainers.

    The parent process never imports the model framework. Each concrete trainer
    launches its training script with the Python interpreter configured for its
    backend in configs/backends.py.
    """

    def __init__(
        self,
        backend_name: str,
        training_script: str | Path,
        *,
        working_directory: str | Path | None = None,
    ) -> None:
        if backend_name not in BACKENDS:
            raise KeyError(
                f"Backend {backend_name!r} is not defined in BACKENDS."
            )

        self.backend_name = backend_name
        self.backend = BACKENDS[backend_name]

        self.python_executable = Path(
            self.backend["python"]
        ).expanduser()

        self.training_script = Path(
            training_script
        ).expanduser().resolve()

        self.working_directory = (
            Path(working_directory).expanduser().resolve()
            if working_directory is not None
            else self.training_script.parent
        )

    def train(self, config: TrainingConfig) -> TrainingResult:
        """
        Run one complete training job and return a standardized result.
        """
        self.validate_config(config)
        self._validate_backend()

        run_name = self.build_run_name(config)
        arguments = list(self.build_arguments(config, run_name))

        command = [
            str(self.python_executable),
            str(self.training_script),
            *arguments,
        ]

        env = self.build_environment(config)

        print("=" * 72)
        print(f"Launching training backend: {self.backend_name}")
        print(f"Python:  {self.python_executable}")
        print(f"Script:  {self.training_script}")
        print(f"Run:     {run_name}")
        print("=" * 72)

        subprocess.run(
            command,
            cwd=self.working_directory,
            env=env,
            check=True,
        )

        return self.collect_result(
            config=config,
            run_name=run_name,
        )

    def build_run_name(self, config: TrainingConfig) -> str:
        """
        Build a deterministic run name.

        This intentionally mirrors the naming convention currently used by
        yolov8_finetune.py so the wrapper can locate the generated run without
        requiring that script to be rewritten first.
        """
        weights_name = Path(str(config.weights)).stem

        parts = [
            config.option,
            f"weights_{weights_name}",
            f"batch_{config.batch}",
            f"lr0_{config.lr0}",
            f"lrf_{config.lrf}",
            f"imgsz_{config.imgsz}",
            f"opt_{config.optimizer}",
        ]

        if config.tag:
            parts.append(config.tag)

        return "_".join(parts)

    def build_environment(
        self,
        config: TrainingConfig,
    ) -> dict[str, str]:
        """
        Build the subprocess environment.

        Concrete trainers can override this if a backend needs additional
        environment variables.
        """
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def validate_config(self, config: TrainingConfig) -> None:
        """Basic validation shared by all trainers."""
        if config.epochs <= 0:
            raise ValueError("epochs must be positive")
        if config.batch == 0:
            raise ValueError("batch cannot be zero")
        if config.imgsz <= 0:
            raise ValueError("imgsz must be positive")
        if config.lr0 <= 0:
            raise ValueError("lr0 must be positive")
        if config.lrf <= 0:
            raise ValueError("lrf must be positive")

        data = Path(config.data).expanduser()
        if not data.is_file():
            raise FileNotFoundError(
                f"Dataset YAML not found: {data}"
            )

    def _validate_backend(self) -> None:
        if not self.python_executable.is_file():
            raise FileNotFoundError(
                f"Python interpreter for backend {self.backend_name!r} "
                f"was not found: {self.python_executable}"
            )

        if not self.training_script.is_file():
            raise FileNotFoundError(
                f"Training script not found: {self.training_script}"
            )

        if not self.working_directory.is_dir():
            raise FileNotFoundError(
                f"Working directory not found: {self.working_directory}"
            )

    @abstractmethod
    def build_arguments(
        self,
        config: TrainingConfig,
        run_name: str,
    ) -> Sequence[str]:
        """
        Translate TrainingConfig into the CLI understood by this backend's
        training script.

        The Python executable and script path are added by Trainer.train().
        """
        raise NotImplementedError

    @abstractmethod
    def collect_result(
        self,
        config: TrainingConfig,
        run_name: str,
    ) -> TrainingResult:
        """
        Inspect the files generated by the backend and return TrainingResult.
        """
        raise NotImplementedError
