from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from config.backends import BACKENDS
from .base_predictor import Predictor


class Yolov8Predictor(Predictor):
    """YOLOv8 predictor that runs inference in the dedicated YOLOv8 environment."""

    def __init__(self, framework="yolov8", models_params=None):
        backend = BACKENDS["yolov8"]
        super().__init__(repo=str(backend.get("repo", "yolov8")), framework=framework)

        self.backend = backend
        self.python_executable = Path(backend["python"]).expanduser()
        self.worker_script = Path(__file__).with_name("yolov8_worker.py")
        self.models_params = list(models_params or [])
        self.images = []

    def load(self, models_params, images_path):
        """Register models and images; actual model loading happens in the worker."""
        self.models_params = list(models_params)
        self.models = list(models_params)
        self.images = [
            str(Path(image).expanduser().resolve())
            for image in self.load_images(images_path)
        ]

        for model_param in self.models_params:
            print(f"Registered YOLOv8 model: {model_param['weight']}")

    def load_one_model(self, model_param):
        """Required by Predictor; process-backed loading is deferred to the worker."""
        self.models.append(model_param)

    def predict_one_model(self, model, image):
        """Required by Predictor; use predict() for process-backed inference."""
        raise RuntimeError(
            "Yolov8Predictor performs inference in yolov8_worker.py. "
            "Call predict() instead."
        )

    def predict(self):
        """Run all YOLOv8 models in one dedicated backend subprocess."""
        if not self.models_params:
            raise ValueError("No YOLOv8 model configurations have been loaded.")
        if not self.images:
            raise ValueError("No images have been loaded.")
        if not self.python_executable.is_file():
            raise FileNotFoundError(
                f"YOLOv8 interpreter not found: {self.python_executable}"
            )
        if not self.worker_script.is_file():
            raise FileNotFoundError(
                f"YOLOv8 worker script not found: {self.worker_script}"
            )

        request = {
            "framework": self.framework,
            "models": self.models_params,
            "images": self.images,
        }

        with tempfile.TemporaryDirectory(prefix="orddc_yolov8_") as temp_dir:
            temp_dir = Path(temp_dir)
            request_path = temp_dir / "request.json"
            output_path = temp_dir / "predictions.json"

            request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

            command = [
                str(self.python_executable),
                str(self.worker_script),
                "--request",
                str(request_path),
                "--output",
                str(output_path),
            ]

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            print("=" * 72)
            print("Launching YOLOv8 backend")
            print(f"Python: {self.python_executable}")
            print(f"Models: {len(self.models_params)}")
            print(f"Images: {len(self.images)}")
            print("=" * 72)

            subprocess.run(command, env=env, check=True)

            if not output_path.is_file():
                raise RuntimeError(
                    "YOLOv8 backend completed without producing predictions."
                )

            result = json.loads(output_path.read_text(encoding="utf-8"))

        if result["images"] != self.images:
            raise RuntimeError(
                "YOLOv8 backend returned predictions in a different image order."
            )

        boxes_list = result["boxes"]
        scores_list = result["scores"]
        labels_list = result["labels"]

        if not (
            len(boxes_list)
            == len(scores_list)
            == len(labels_list)
            == len(self.models_params)
        ):
            raise RuntimeError(
                "YOLOv8 backend returned an unexpected number of model outputs."
            )

        return boxes_list, scores_list, labels_list