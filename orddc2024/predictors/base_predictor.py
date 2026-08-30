from __future__ import annotations

from abc import ABC, abstractmethod
import os

from PIL import Image

from orddc2024.predictions.prediction_result import PredictionResult


class Predictor(ABC):
    """
    Base predictor.

    Canonical output:
        list[PredictionResult]

    Each PredictionResult represents one model over the complete image set,
    using normalized XYXY boxes and zero-based class IDs.
    """

    def __init__(self, repo, framework):
        self.repo = repo
        self.framework = framework
        self.models = []
        self.images: list[str] = []

    def load(self, weights, images_path):
        self.weights = weights
        self.images = self.load_images(images_path)

        for weight in weights:
            print(f"Loading model from weight: {weight}")
            self.load_one_model(weight)

    def load_images(self, images_path):
        if os.path.isdir(images_path):
            return [
                os.path.join(images_path, img)
                for img in sorted(os.listdir(images_path))
                if img.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

        if os.path.isfile(images_path):
            with open(images_path, "r", encoding="utf-8") as file:
                return [
                    line.strip()
                    for line in file.readlines()
                    if line.strip()
                ]

        raise ValueError(
            "Invalid path to images. Provide a directory or a text file "
            "with image paths."
        )

    @abstractmethod
    def load_one_model(self, weight):
        pass

    @abstractmethod
    def predict_one_model(self, model, image):
        pass

    def predict(self) -> list[PredictionResult]:
        results: list[PredictionResult] = []

        for model_index, model in enumerate(self.models):
            model_boxes = []
            model_scores = []
            model_labels = []

            for image in self.images:
                boxes, scores, labels = self.predict_one_model(model, image)
                model_boxes.append(boxes)
                model_scores.append(scores)
                model_labels.append(labels)

            results.append(
                PredictionResult(
                    images=list(self.images),
                    boxes=model_boxes,
                    scores=model_scores,
                    labels=model_labels,
                    metadata={
                        "framework": self.framework,
                        "repo": self.repo,
                        "model_index": model_index,
                    },
                )
            )

        return results

    def predict_legacy(self):
        """
        Compatibility adapter for older ORDDC ensemble code.

        Labels remain zero-based under the new canonical contract.
        """
        results = self.predict()
        return (
            [result.boxes for result in results],
            [result.scores for result in results],
            [result.labels for result in results],
        )

    @staticmethod
    def normalize_box(box, img_width, img_height):
        x1, y1, x2, y2 = box
        return [
            x1 / img_width,
            y1 / img_height,
            x2 / img_width,
            y2 / img_height,
        ]

    @staticmethod
    def denormalize_box(box, img_width, img_height):
        x1, y1, x2, y2 = box
        return [
            x1 * img_width,
            y1 * img_height,
            x2 * img_width,
            y2 * img_height,
        ]

    def get_image_size(self, image_path):
        with Image.open(image_path) as img:
            return img.size