from .base_predictor import Predictor
# from .ultralytics_predictor import UltralyticsPredictor
# from .yolov5_predictor import Yolov5Predictor
from ultralytics import YOLO

import os
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence
from ensemble_boxes import weighted_boxes_fusion, nms, non_maximum_weighted, soft_nms

class EnsemblePredictor(Predictor):
    """
    Run multiple custom Predictors and fuse their detections image-by-image.

    Canonical return shape:
        boxes[image_index][detection_index]  -> [x1, y1, x2, y2]
        scores[image_index][detection_index] -> float
        labels[image_index][detection_index] -> int

    Child predictors must return normalized XYXY boxes and must process the same
    images in the same order.
    """
    
    _RESERVED_KEYS = {
        "name",
        "framework",
        "predictor_params",
        "ensemble_weight",
        "output_has_model_axis",
        "label_map",
    }
    
    def __init__(
        self,
        framework: str,
        models_params: Sequence[Mapping[str, Any]],
        *,
        predictor_registry: Mapping[str, type[Predictor]],
        fusion_method: str = "wbf",
        iou_thr: float = 0.55,
        skip_box_thr: float = 0.001,
        sigma: float = 0.1,
        conf_type: str = "avg",
        allows_overflow: bool = False,
        max_workers: int | None = None,
        strict_image_order: bool = True,
    ):
        super().__init__("ensemble", framework)
        
        self.models_params = list(models_params)
        self.predictor_registry = dict(predictor_registry)
        
        self.fusion_method = fusion_method.lower()
        self.iou_thr = float(iou_thr)
        self.skip_box_thr = float(skip_box_thr)
        self.sigma = float(sigma)
        self.conf_type = conf_type
        self.allows_overflow = bool(allows_overflow)
        self.max_workers = max_workers
        self.strict_image_order = strict_image_order
        
        self.images: list[str] = []
        self.members: list[dict[str, Any]] = []
        self.raw_predictions = None
        self.last_predictions = None

        if self.fusion_method not in {"wbf", "nms", "nmw", "soft_nms"}:
            raise ValueError(
                "fusion_method must be one of: wbf, nms, nmw, soft_nms"
            )
        if not 0 <= self.iou_thr <= 1:
            raise ValueError("iou_thr must be in [0, 1]")
        if not 0 <= self.skip_box_thr <= 1:
            raise ValueError("skip_box_thr must be in [0, 1]")
    
    def load_one_model(self, weight):
        raise NotImplementedError(
            "EnsemblePredictor does not implement load_one_model. "
            "Use child predictors instead."
        )
    
    def _load_member(
        self,
        index: int,
        config: Mapping[str, Any],
        images_path: str,
    ) -> dict[str, Any]:
        framework = str(config.get("framework", "")).strip()
        if framework not in self.predictor_registry:
            raise ValueError(
                f"Model {index}: no predictor registered for {framework!r}. "
                f"Registered frameworks: {sorted(self.predictor_registry)}"
            )
        # Accept either:
        #   {"framework": "...", "predictor_params": {...}}
        # or a flat leaderboard-style model dictionary.
        if "predictor_params" in config:
            predictor_params = config["predictor_params"]
            if not isinstance(predictor_params, Mapping):
                raise TypeError("predictor_params must be a mapping")
            model_param = dict(predictor_params)
        else:
            model_param = {
                key: value
                for key, value in config.items()
                if key not in self._RESERVED_KEYS
            }
        weight_path = model_param.get("weight")
        if weight_path and not Path(str(weight_path)).exists():
            raise FileNotFoundError(
                f"Model {index} weight does not exist: {weight_path}"
            )
        ensemble_weight = float(config.get("ensemble_weight", 1.0))
        if ensemble_weight <= 0:
            raise ValueError("ensemble_weight must be positive")
        predictor_class = self.predictor_registry[framework]
        predictor = predictor_class(framework, [model_param])
        predictor.load([model_param], images_path)
        child_images = [str(path) for path in predictor.images]
        if len(child_images) != len(self.images):
            raise ValueError(
                f"{framework}: expected {len(self.images)} images, "
                f"loaded {len(child_images)}"
            )
        if self.strict_image_order and child_images != self.images:
            raise ValueError(
                f"{framework}: image ordering differs from the ensemble order. "
                "Fusion would combine detections from different images."
            )
        label_map = config.get("label_map")
        if label_map is not None:
            if not isinstance(label_map, Mapping):
                raise TypeError("label_map must be a mapping")
            label_map = {
                int(source): int(target)
                for source, target in label_map.items()
            }
        return {
            "index": index,
            "name": config.get("name", f"{framework}_{index}"),
            "predictor": predictor,
            "ensemble_weight": ensemble_weight,
            # True for the UltralyticsPredictor shown in the question.
            "output_has_model_axis": bool(
                config.get("output_has_model_axis", True)
            ),
            "label_map": label_map,
        }
        
    def load_images(self, images_path):
        if os.path.isdir(images_path):
            return [os.path.join(images_path, img) for img in os.listdir(images_path) if img.endswith('.jpg') or img.endswith('.png')]
        elif os.path.isfile(images_path):
            with open(images_path, 'r') as file:
                return [line.strip() for line in file.readlines()]
        else:
            raise ValueError("Invalid path to images. Provide a directory or a text file with image paths.")
        
    def load(
        self,
        models_params: Sequence[Mapping[str, Any]] | None,
        images_path: str | Path,
    ):
        """
        Load the shared image list and each child predictor.

        Pass models_params=None to reuse the configuration supplied to __init__.
        """
        if models_params is not None:
            self.models_params = list(models_params)
        if not self.models_params:
            raise ValueError("At least one ensemble member is required")

        self.images = [str(path) for path in self.load_images(str(images_path))]
        if not self.images:
            raise ValueError(f"No images found at {images_path}")

        self.members.clear()

        # Load sequentially. Concurrent loading can cause temporary GPU-memory
        # spikes and CUDA initialization races. Prediction can still be threaded.
        for index, config in enumerate(self.models_params):
            member = self._load_member(index, config, str(images_path))
            self.members.append(member)
    
    def _predict_member(self, member: Mapping[str, Any], batch_size: int):
        try:
            raw = member["predictor"].predict(batch_size=batch_size)
            boxes, scores, labels = self._normalize_child_output(
                raw,
                output_has_model_axis=member["output_has_model_axis"],
                member_name=str(member["name"]),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Prediction failed for ensemble member {member['name']!r}"
            ) from exc

        label_map = member["label_map"]
        if label_map:
            labels = [
                [label_map.get(label, label) for label in image_labels]
                for image_labels in labels
            ]

        return boxes, scores, labels

    def _normalize_child_output(
        self,
        raw,
        *,
        output_has_model_axis: bool,
        member_name: str,
    ):
        """
        Normalize the output of a child predictor to ensure it has the expected format.

        Args:
            raw: The raw output from the child predictor.
            output_has_model_axis: Whether the output includes a model axis.
            member_name: The name of the ensemble member.

        Returns:
            A tuple of (boxes, scores, labels) with cleaned and validated data.
        """
        if not isinstance(raw, (tuple, list)) or len(raw) != 3:
            raise TypeError(
                f"{member_name}: predict() must return boxes, scores, labels"
            )

        boxes, scores, labels = raw

        # The supplied UltralyticsPredictor returns:
        #   [boxes_per_image], [scores_per_image], [labels_per_image]
        if output_has_model_axis:
            if len(boxes) != 1 or len(scores) != 1 or len(labels) != 1:
                raise ValueError(
                    f"{member_name}: expected singleton model-axis output"
                )
            boxes, scores, labels = boxes[0], scores[0], labels[0]

        if not (
            len(boxes)
            == len(scores)
            == len(labels)
            == len(self.images)
        ):
            raise ValueError(
                f"{member_name}: prediction count does not match image count"
            )

        clean_boxes, clean_scores, clean_labels = [], [], []

        for image_index, (image_boxes, image_scores, image_labels) in enumerate(
            zip(boxes, scores, labels)
        ):
            b = np.asarray(image_boxes, dtype=np.float64)
            s = np.asarray(image_scores, dtype=np.float64).reshape(-1)
            l = np.asarray(image_labels).reshape(-1)

            b = (
                np.empty((0, 4), dtype=np.float64)
                if b.size == 0
                else b.reshape(-1, 4)
            )

            if not (len(b) == len(s) == len(l)):
                raise ValueError(
                    f"{member_name}, image {image_index}: "
                    "boxes/scores/labels lengths differ"
                )

            finite = np.isfinite(b).all(axis=1) & np.isfinite(s)
            b = np.clip(b, 0.0, 1.0)
            positive_area = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
            valid_score = (s >= 0.0) & (s <= 1.0)
            keep = finite & positive_area & valid_score

            clean_boxes.append(b[keep].tolist())
            clean_scores.append(s[keep].astype(float).tolist())
            clean_labels.append(l[keep].astype(np.int64).tolist())

        return clean_boxes, clean_scores, clean_labels
    
    def _fuse_predictions(self, raw_predictions):
        weights = [
            member["ensemble_weight"]
            for member in self.members
        ]

        output_boxes, output_scores, output_labels = [], [], []

        for image_index in range(len(self.images)):
            boxes_list = [
                prediction[0][image_index]
                for prediction in raw_predictions
            ]
            scores_list = [
                prediction[1][image_index]
                for prediction in raw_predictions
            ]
            labels_list = [
                prediction[2][image_index]
                for prediction in raw_predictions
            ]

            boxes, scores, labels = self._fuse_one_image(
                boxes_list,
                scores_list,
                labels_list,
                weights,
            )
            output_boxes.append(boxes)
            output_scores.append(scores)
            output_labels.append(labels)

        return output_boxes, output_scores, output_labels

    def _fuse_one_image(
        self,
        boxes_list,
        scores_list,
        labels_list,
        weights,
    ):
        if not any(boxes_list):
            return [], [], []

        common = {
            "weights": weights,
            "iou_thr": self.iou_thr,
        }

        if self.fusion_method == "wbf":
            boxes, scores, labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                skip_box_thr=self.skip_box_thr,
                conf_type=self.conf_type,
                allows_overflow=self.allows_overflow,
                **common,
            )
        elif self.fusion_method == "nms":
            boxes, scores, labels = nms(
                boxes_list,
                scores_list,
                labels_list,
                **common,
            )
        elif self.fusion_method == "nmw":
            boxes, scores, labels = non_maximum_weighted(
                boxes_list,
                scores_list,
                labels_list,
                skip_box_thr=self.skip_box_thr,
                **common,
            )
        else:
            boxes, scores, labels = soft_nms(
                boxes_list,
                scores_list,
                labels_list,
                sigma=self.sigma,
                thresh=self.skip_box_thr,
                **common,
            )

        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)

        # Deterministic descending-confidence order for evaluation/CSV output.
        order = np.argsort(-scores, kind="stable")

        return (
            boxes[order].tolist(),
            scores[order].astype(float).tolist(),
            labels[order].astype(int).tolist(),
        )
        
    def predict_one_model(self, model, image):
        raise NotImplementedError(
            "EnsemblePredictor does not implement _predict_one_model. "
            "Use child predictors instead."
        )
    
    def predict(self, batch_size: int = 128):
        """
        Return fused predictions without a model axis:

            boxes_per_image, scores_per_image, labels_per_image
        """
        if not self.members:
            raise RuntimeError("Call load(...) before predict(...)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        worker_count = (
            len(self.members)
            if self.max_workers is None
            else min(self.max_workers, len(self.members))
        )
        if worker_count <= 0:
            raise ValueError("max_workers must be positive or None")

        # executor.map preserves the configured member order.
        if worker_count == 1:
            raw = [
                self._predict_member(member, batch_size)
                for member in self.members
            ]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                raw = list(
                    executor.map(
                        lambda member: self._predict_member(member, batch_size),
                        self.members,
                    )
                )

        fused = self._fuse_predictions(raw)
        self.raw_predictions = raw
        self.last_predictions = fused
        return fused
    