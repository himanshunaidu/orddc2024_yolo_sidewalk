# YOLO26 predictor for the ORDDC2024 project. Uses yolo26_worker.py for inference.
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from config.backends import BACKENDS
from orddc2024.predictions.prediction_result import PredictionResult

from .base_predictor import Predictor


class Yolo26Predictor(Predictor):
    def __init__(self, framework: str = "yolo26", models_params=None):
        backend = BACKENDS["yolo26"]

        super().__init__(
            repo=str(backend.get("repo", "yolo26")),
            framework=framework,
        )

        self.backend = backend
        self.python_executable = Path(backend["python"]).expanduser()
        self.worker_script = Path(__file__).with_name("yolo26_worker.py")
        self.models_params = list(models_params or [])

    def load(self, models_params, images_path):
        self.models_params = list(models_params)
        self.models = list(models_params)
        self.images = [
            str(Path(image).expanduser().resolve())
            for image in self.load_images(images_path)
        ]

        for model_param in self.models_params:
            print(f"Registered YOLO26 model: {model_param['weight']}")

    def load_one_model(self, model_param):
        self.models.append(model_param)

    def predict_one_model(self, model, image):
        raise RuntimeError(
            "Yolo26Predictor uses yolo26_worker.py. Call predict()."
        )

    def predict(self) -> list[PredictionResult]:
        if not self.models_params:
            raise ValueError("No YOLO26 model configurations have been loaded.")
        if not self.images:
            raise ValueError("No images have been loaded.")
        if not self.python_executable.is_file():
            raise FileNotFoundError(
                f"YOLO26 Python interpreter not found: {self.python_executable}"
            )
        if not self.worker_script.is_file():
            raise FileNotFoundError(
                f"YOLO26 worker script not found: {self.worker_script}"
            )

        request = {
            "framework": self.framework,
            "models": self.models_params,
            "images": self.images,
        }

        with tempfile.TemporaryDirectory(prefix="orddc_yolo26_") as temp_dir:
            temp_dir = Path(temp_dir)
            request_path = temp_dir / "request.json"
            output_path = temp_dir / "predictions.json"

            request_path.write_text(
                json.dumps(request, indent=2),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    str(self.python_executable),
                    str(self.worker_script),
                    "--request",
                    str(request_path),
                    "--output",
                    str(output_path),
                ],
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                check=True,
            )

            if not output_path.is_file():
                raise RuntimeError(
                    "YOLO26 backend completed without creating predictions."
                )

            worker_result = json.loads(
                output_path.read_text(encoding="utf-8")
            )

        if worker_result["images"] != self.images:
            raise RuntimeError(
                "YOLO26 backend returned predictions in a different image order."
            )

        boxes_list = worker_result["boxes"]
        scores_list = worker_result["scores"]
        labels_list = worker_result["labels"]

        if not (
            len(boxes_list)
            == len(scores_list)
            == len(labels_list)
            == len(self.models_params)
        ):
            raise RuntimeError(
                "YOLO26 backend returned an unexpected number of model outputs."
            )

        return [
            PredictionResult(
                images=list(self.images),
                boxes=boxes,
                scores=scores,
                labels=labels,
                metadata={
                    "backend": "yolo26",
                    "framework": self.framework,
                    "repo": self.repo,
                    "model_index": model_index,
                    "weight": model_param["weight"],
                    "inference": dict(model_param),
                },
            )
            for model_index, (model_param, boxes, scores, labels)
            in enumerate(
                zip(
                    self.models_params,
                    boxes_list,
                    scores_list,
                    labels_list,
                )
            )
        ]