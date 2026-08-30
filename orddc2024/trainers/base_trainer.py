from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from config.backends import BACKENDS
from ..predictions.prediction_result import PredictionResult


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

    Training artifacts and prediction artifacts deliberately remain separate:
    `PredictionResult` owns prediction contents/formats, while TrainingResult
    merely provides a convenient place to cache and discover predictions that
    belong to this training run.
    """

    backend: str
    run_name: str
    run_dir: Path

    best_weights: Path | None = None
    last_weights: Path | None = None

    results_file: Path | None = None
    metadata_file: Path | None = None

    predictions_dir: Path | None = None
    prediction_caches: dict[str, Path] = field(default_factory=dict)

    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)

        if self.best_weights is not None:
            self.best_weights = Path(self.best_weights)

        if self.last_weights is not None:
            self.last_weights = Path(self.last_weights)

        if self.results_file is not None:
            self.results_file = Path(self.results_file)

        if self.metadata_file is not None:
            self.metadata_file = Path(self.metadata_file)

        if self.predictions_dir is None:
            self.predictions_dir = self.run_dir / "predictions"
        else:
            self.predictions_dir = Path(self.predictions_dir)

        self.prediction_caches = {
            str(name): Path(path)
            for name, path in self.prediction_caches.items()
        }

    def cache_prediction(
        self,
        name: str,
        prediction: PredictionResult,
        *,
        overwrite: bool = True,
        save_orddc_folder: bool = False,
        dataset_root: str | Path | None = None,
        include_scores: bool = True,
    ) -> Path:
        """
        Save a PredictionResult under this training run.

        Example:
            result.cache_prediction("val", prediction)

        produces:
            <run_dir>/predictions/val.npz

        Optionally also creates:
            <run_dir>/predictions/val_txt/

        using PredictionResult.save_orddc_folder().
        """
        cache_name = self._validate_cache_name(name)

        assert self.predictions_dir is not None
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        cache_path = self.predictions_dir / f"{cache_name}.npz"

        if cache_path.exists() and not overwrite:
            raise FileExistsError(
                f"Prediction cache already exists: {cache_path}"
            )

        prediction.save_npz(cache_path)
        self.prediction_caches[cache_name] = cache_path

        if save_orddc_folder:
            if dataset_root is None:
                raise ValueError(
                    "dataset_root is required when save_orddc_folder=True"
                )

            prediction.save_orddc_folder(
                self.predictions_dir / f"{cache_name}_txt",
                dataset_root=dataset_root,
                include_scores=include_scores,
            )

        return cache_path

    def load_prediction(
        self,
        name: str,
    ) -> PredictionResult:
        """
        Load a cached PredictionResult associated with this training run.
        """
        cache_name = self._validate_cache_name(name)

        cache_path = self.prediction_caches.get(cache_name)

        if cache_path is None:
            assert self.predictions_dir is not None
            candidate = self.predictions_dir / f"{cache_name}.npz"

            if candidate.is_file():
                cache_path = candidate
                self.prediction_caches[cache_name] = candidate

        if cache_path is None or not cache_path.is_file():
            raise FileNotFoundError(
                f"No cached prediction named {cache_name!r} exists for "
                f"training run {self.run_name!r}."
            )

        return PredictionResult.load_npz(cache_path)

    def refresh_prediction_caches(self) -> dict[str, Path]:
        """
        Re-scan <run_dir>/predictions for .npz caches.

        This is useful when prediction files were created or regenerated in a
        separate process after the TrainingResult object was first created.
        """
        assert self.predictions_dir is not None

        if not self.predictions_dir.is_dir():
            self.prediction_caches = {}
            return self.prediction_caches

        self.prediction_caches = {
            path.stem: path
            for path in sorted(self.predictions_dir.glob("*.npz"))
            if path.is_file()
        }

        return dict(self.prediction_caches)

    def has_prediction(self, name: str) -> bool:
        """
        Return True if the named .npz prediction cache exists.
        """
        cache_name = self._validate_cache_name(name)

        cache_path = self.prediction_caches.get(cache_name)
        if cache_path is not None and cache_path.is_file():
            return True

        assert self.predictions_dir is not None
        candidate = self.predictions_dir / f"{cache_name}.npz"

        if candidate.is_file():
            self.prediction_caches[cache_name] = candidate
            return True

        return False

    @staticmethod
    def _validate_cache_name(name: str) -> str:
        """
        Keep cache names simple so all caches stay directly under predictions/.
        """
        name = str(name).strip()

        if not name:
            raise ValueError("Prediction cache name cannot be empty")

        if Path(name).name != name:
            raise ValueError(
                "Prediction cache name must be a simple file name "
                "without directory components"
            )

        if name.endswith(".npz"):
            name = name[:-4]

        if not name:
            raise ValueError("Prediction cache name cannot be empty")

        return name


class Trainer(ABC):
    """
    Base class for process-backed model trainers.

    The parent process never imports the model framework. Each concrete trainer
    launches its training script with the Python interpreter configured for its
    backend in config/backends.py.
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

        result = self.collect_result(
            config=config,
            run_name=run_name,
        )

        # Pick up any prediction caches that may already exist, for example
        # when re-opening an existing run with exist_ok=True.
        result.refresh_prediction_caches()

        return result

    def build_run_name(self, config: TrainingConfig) -> str:
        """
        Build a deterministic run name.
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