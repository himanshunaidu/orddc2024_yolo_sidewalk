from __future__ import annotations

from typing import Type

from .predictors.base_predictor import Predictor
from .predictors.yolov8_predictor import Yolov8Predictor
from .predictors.yolo26_predictor import Yolo26Predictor

from .trainers.base_trainer import Trainer
from .trainers.yolov8_trainer import YoloV8Trainer
from .trainers.yolo26_trainer import Yolo26Trainer


TRAINER_REGISTRY: dict[str, Type[Trainer]] = {
    "yolov8": YoloV8Trainer,
    "yolo26": Yolo26Trainer,
}

PREDICTOR_REGISTRY: dict[str, Type[Predictor]] = {
    "yolov8": Yolov8Predictor,
    "yolo26": Yolo26Predictor,
}


def get_trainer(
    backend_name: str,
    **kwargs,
) -> Trainer:
    """
    Create the trainer registered for a backend.

    Example:
        trainer = get_trainer("yolov8")
    """
    try:
        trainer_class = TRAINER_REGISTRY[backend_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown trainer backend {backend_name!r}. "
            f"Available backends: {sorted(TRAINER_REGISTRY)}"
        ) from exc

    return trainer_class(**kwargs)


def get_predictor(
    backend_name: str,
    **kwargs,
) -> Predictor:
    """
    Create the predictor registered for a backend.

    Example:
        predictor = get_predictor("yolo26")
    """
    try:
        predictor_class = PREDICTOR_REGISTRY[backend_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown predictor backend {backend_name!r}. "
            f"Available backends: {sorted(PREDICTOR_REGISTRY)}"
        ) from exc

    return predictor_class(**kwargs)


def available_trainer_backends() -> tuple[str, ...]:
    return tuple(sorted(TRAINER_REGISTRY))


def available_predictor_backends() -> tuple[str, ...]:
    return tuple(sorted(PREDICTOR_REGISTRY))


def available_backends() -> tuple[str, ...]:
    """
    Return backends that have both a trainer and predictor registered.
    """
    return tuple(
        sorted(
            set(TRAINER_REGISTRY)
            & set(PREDICTOR_REGISTRY)
        )
    )