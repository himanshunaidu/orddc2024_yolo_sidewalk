from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from config.backends import BACKENDS

from .base_predictor import Predictor


class Yolo26Predictor(Predictor):
    """
    YOLO26 predictor backed by a dedicated Python environment.

    Ultralytics is deliberately NOT imported in this module. Inference is
    delegated to `yolo26_worker.py`, launched with the interpreter configured
    in BACKENDS["yolo26"]["python"].

    Output format matches the existing ORDDC predictors:

        boxes_list[model_idx][image_idx][detection_idx] = [x1, y1, x2, y2]
        scores_list[model_idx][image_idx][detection_idx] = confidence
        labels_list[model_idx][image_idx][detection_idx] = class_id

    Boxes are normalized XYXY. By default labels remain one-based to preserve
    compatibility with the original ORDDC inference code. Set
    `label_offset=0` in a model config if zero-based labels are desired.
    """

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
        self.images: list[str] = []

    def load(self, models_params, images_path):
        """
        Store model configurations and image paths.

        Models are intentionally not loaded in the parent process. Actual
        loading happens inside the YOLO26 backend subprocess.
        """
        self.models_params = list(models_params)
        self.models = list(models_params)
        self.images = [
            str(Path(image).expanduser().resolve())
            for image in self.load_images(images_path)
        ]

        for model_param in self.models_params:
            print(f"Registered YOLO26 model: {model_param['weight']}")

    def load_one_model(self, model_param):
        """
        Required by the Predictor interface.

        For process-backed predictors, loading is deferred to the worker.
        """
        self.models.append(model_param)

    def predict_one_model(self, model, image):
        """
        Required by the Predictor interface.

        Per-model inference happens in the backend worker rather than in the
        parent process, so this method should never be called directly.
        """
        raise RuntimeError(
            "Yolo26Predictor performs inference through yolo26_worker.py. "
            "Call predict() instead of predict_one_model()."
        )

    def predict(self):
        """
        Run all registered YOLO26 models in one backend subprocess.

        A single subprocess imports the configured YOLO26-compatible
        Ultralytics installation and evaluates every configured YOLO26 model.
        This avoids one process launch per model while still isolating the
        Ultralytics version from the parent process.
        """
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
            print("Launching YOLO26 backend")
            print(f"Python: {self.python_executable}")
            print(f"Worker: {self.worker_script}")
            print(f"Models: {len(self.models_params)}")
            print(f"Images: {len(self.images)}")
            print("=" * 72)

            subprocess.run(
                command,
                env=env,
                check=True,
            )

            if not output_path.is_file():
                raise RuntimeError(
                    "YOLO26 backend completed without creating a prediction file."
                )

            result = json.loads(output_path.read_text(encoding="utf-8"))

        worker_images = result["images"]
        if worker_images != self.images:
            raise RuntimeError(
                "YOLO26 backend returned predictions in a different image order."
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
                "YOLO26 backend returned an unexpected number of model outputs."
            )

        return boxes_list, scores_list, labels_list