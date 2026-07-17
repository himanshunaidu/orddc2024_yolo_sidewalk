from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from PIL import Image
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, RANK, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import plot_images

BoxFormat = Literal["xyxy", "xyxyn", "xywh", "xywhn"]
AnnotationRecord = Mapping[str, Any]
RecordCollection = Sequence[AnnotationRecord] | Mapping[str | Path, AnnotationRecord]

@dataclass(slots=True)
class ValidationResult:
    """Results returned by CustomValidator.evaluate()."""

    overall: dict[str, float]
    # per_class: list[dict[str, Any]]
    # per_image: dict[str, dict[str, float | int]]
    confusion_matrix: np.ndarray
    metrics: DetMetrics
    
class CustomValidator(BaseValidator):
    """
    Evaluate already-postprocessed object-detection predictions.

    This class does not load or run a model. It reuses Ultralytics'
    matching, DetMetrics, ConfusionMatrix, and plotting behavior.

    Ground-truth record format:
        {"bboxes": array-like [N, 4], "cls": array-like [N]}

    Prediction record format:
        {
            "bboxes": array-like [M, 4],
            "conf": array-like [M],
            "cls": array-like [M],
        }

    Records may be sequences aligned with `images`, or mappings keyed by full
    image path, filename, or stem. Internally, every box is converted to
    normalized XYXY coordinates before matching.
    """
    def __init__(
        self,
        names: Sequence[str] | Mapping[int, str],
        *,
        save_dir: str | Path = "runs/custom_val",
        plots: bool = True,
        verbose: bool = True,
        single_cls: bool = False,
        min_conf: float = 0.001,
        max_det: int = 300,
        confusion_conf: float = 0.25,
        confusion_iou: float = 0.45,
        device: str | torch.device = "cpu",
    ) -> None:
        if not 0.0 <= min_conf <= 1.0:
            raise ValueError("min_conf must be in [0, 1]")
        if not 0.0 <= confusion_conf <= 1.0:
            raise ValueError("confusion_conf must be in [0, 1]")
        if not 0.0 <= confusion_iou <= 1.0:
            raise ValueError("confusion_iou must be in [0, 1]")
        if max_det <= 0:
            raise ValueError("max_det must be positive")
        
        save_dir = Path(save_dir)
        super().__init__(
            dataloader=None,
            save_dir=save_dir,
            args={
                "task": "detect",
                "plots": plots,
                "verbose": verbose,
                "single_cls": single_cls,
                "save_json": False,
                "save_txt": False,
                "visualize": False,
            },
        )
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.training = False
        self.names = self._normalize_names(names)
        self.nc = len(self.names)
        self.iouv = torch.linspace(0.50, 0.95, 10, device=self.device)
        self.niou = int(self.iouv.numel())
        self.min_conf = float(min_conf)
        self.max_det = int(max_det)
        self.confusion_conf = float(confusion_conf)
        self.confusion_iou = float(confusion_iou)
        self.reset()
        
    def evaluate(
        self,
        images: Sequence[str | Path],
        ground_truths: RecordCollection,
        predictions: RecordCollection,
        *,
        ground_truth_box_format: BoxFormat = "xywhn",
        prediction_box_format: BoxFormat = "xyxyn",
        ground_truth_label_offset: int = 0,
        prediction_label_offset: int = 0,
    ) -> ValidationResult:
        """Evaluate predictions against ground truth."""
        image_paths = [Path(path) for path in images]
        if not image_paths:
            raise ValueError("images must contain at least one image")

        self._validate_collection(ground_truths, len(image_paths), "ground_truths")
        self._validate_collection(predictions, len(image_paths), "predictions")
        self.reset()
        total_targets = 0
        
        tp_parts: list[np.ndarray] = []
        target_boxes_parts: list[np.ndarray] = []
        target_cls_parts: list[np.ndarray] = []
        conf_parts: list[np.ndarray] = []
        pred_boxes_parts: list[np.ndarray] = []
        pred_cls_parts: list[np.ndarray] = []
        target_cls_parts: list[np.ndarray] = []
        
        target_boxes_tensor_parts: list[torch.Tensor] = []
        target_cls_tensor_parts: list[torch.Tensor] = []
        pred_boxes_tensor_parts: list[torch.Tensor] = []
        pred_conf_tensor_parts: list[torch.Tensor] = []
        pred_cls_tensor_parts: list[torch.Tensor] = []

        for image_index, image_path in enumerate(image_paths):
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")

            with Image.open(image_path) as image:
                image_width, image_height = image.size

            gt_record = self._resolve_record(
                ground_truths, image_path, image_index, "ground_truths"
            )
            pred_record = self._resolve_record(
                predictions, image_path, image_index, "predictions"
            )

            target_boxes, target_cls = self._prepare_ground_truth(
                gt_record,
                box_format=ground_truth_box_format,
                image_width=image_width,
                image_height=image_height,
                label_offset=ground_truth_label_offset,
                image_name=image_path.name,
            )
            pred_boxes, pred_cls, pred_conf = self._prepare_prediction(
                pred_record,
                box_format=prediction_box_format,
                image_width=image_width,
                image_height=image_height,
                label_offset=prediction_label_offset,
                image_name=image_path.name,
            )
            print(pred_boxes, pred_conf, pred_cls, target_boxes, target_cls)
            tp = self._process_batch(
                pred_boxes, pred_conf, pred_cls, target_boxes, target_cls
            )

            total_targets += int(target_cls.numel())
            # self.update_metrics(prediction, target, image_name=image_path.name)
            
            tp_parts.append(tp)
            target_boxes_parts.append(target_boxes.cpu().numpy())
            conf_parts.append(pred_conf.cpu().numpy())
            pred_boxes_parts.append(pred_boxes.cpu().numpy())
            pred_cls_parts.append(pred_cls.cpu().numpy())
            target_cls_parts.append(target_cls.cpu().numpy())
            
            target_boxes_tensor_parts.append(target_boxes)
            target_cls_tensor_parts.append(target_cls)
            pred_boxes_tensor_parts.append(pred_boxes)
            pred_conf_tensor_parts.append(pred_conf)
            pred_cls_tensor_parts.append(pred_cls)
        
        tp = np.concatenate(tp_parts, axis=0)
        target_boxes = np.concatenate(target_boxes_parts, axis=0)
        conf = np.concatenate(conf_parts, axis=0)
        pred_boxes = np.concatenate(pred_boxes_parts, axis=0)
        pred_cls = np.concatenate(pred_cls_parts, axis=0)
        target_cls = np.concatenate(target_cls_parts, axis=0)
        
        target_boxes_tensor = torch.cat(target_boxes_tensor_parts, dim=0)
        pred_boxes_tensor = torch.cat(pred_boxes_tensor_parts, dim=0)
        pred_conf_tensor = torch.cat(pred_conf_tensor_parts, dim=0)
        pred_cls_tensor = torch.cat(pred_cls_tensor_parts, dim=0)
        target_cls_tensor = torch.cat(target_cls_tensor_parts, dim=0)

        if total_targets == 0:
            raise ValueError(
                "No ground-truth objects were found. Detection AP cannot be "
                "calculated without target annotations."
            )

        self.process_metrics(
            tp=tp, conf=conf, pred_cls=pred_cls, target_cls=target_cls,
            target_boxes_tensor=target_boxes_tensor, target_cls_tensor=target_cls_tensor,
            pred_boxes_tensor=pred_boxes_tensor, pred_conf_tensor=pred_conf_tensor, pred_cls_tensor=pred_cls_tensor
        )
        # overall = self.get_stats()
        self.finalize_metrics()
        # self.print_results()

        return ValidationResult(
            overall=dict(self.metrics.results_dict),
            # per_class=self.metrics.summary(),
            # per_image=dict(getattr(self.metrics.box, "image_metrics", {})),
            confusion_matrix=self.confusion_matrix.matrix.copy(),
            metrics=self.metrics,
        )
    
    def evaluate_predictor(
        self,
        predictor: Any,
        ground_truths: RecordCollection,
        *,
        batch_size: int = 128,
        ground_truth_box_format: BoxFormat = "xywhn",
        prediction_box_format: BoxFormat = "xyxyn",
        ground_truth_label_offset: int = 0,
        prediction_label_offset: int = 0,
    ) -> ValidationResult:
        """Run a custom Predictor/EnsemblePredictor and evaluate its output."""
        boxes, scores, labels = predictor.predict(batch_size=batch_size)
        predictions = self.predictions_from_lists(
            predictor.images, boxes, scores, labels
        )
        return self.evaluate(
            images=predictor.images,
            ground_truths=ground_truths,
            predictions=predictions,
            ground_truth_box_format=ground_truth_box_format,
            prediction_box_format=prediction_box_format,
            ground_truth_label_offset=ground_truth_label_offset,
            prediction_label_offset=prediction_label_offset,
        )
        
    def reset(self) -> None:
        """Reset accumulated state before a validation run."""
        self.seen = 0
        self.metrics = DetMetrics(names=self.names)
        self.metrics.names = self.names
        # self.metrics.clear_stats()
        if hasattr(self.metrics, "clear_image_metrics"):
            self.metrics.clear_image_metrics()
        self.confusion_matrix = ConfusionMatrix(
            # names=self.names,
            nc=self.nc,
            task="detect",
            conf=self.confusion_conf,
            iou_thres=self.confusion_iou,
            # save_matches=False,
        )
        self.plots = {}
    
    def process_metrics(
        self,
        tp: np.ndarray,
        conf: np.ndarray,
        pred_cls: np.ndarray,
        target_cls: np.ndarray,
        target_boxes_tensor: torch.Tensor,
        target_cls_tensor: torch.Tensor,
        pred_boxes_tensor: torch.Tensor,
        pred_conf_tensor: torch.Tensor,
        pred_cls_tensor: torch.Tensor
    ) -> None:
        """Process metrics for a batch of predictions and targets."""
        self.metrics.process(tp=tp, conf=conf, pred_cls=pred_cls, target_cls=target_cls)
        # def process_batch(self, detections, gt_bboxes, gt_cls):
        # """
        # Update confusion matrix for object detection task.

        # Args:
        #     detections (Array[N, 6] | Array[N, 7]): Detected bounding boxes and their associated information.
        #                               Each row should contain (x1, y1, x2, y2, conf, class)
        #                               or with an additional element `angle` when it's obb.
        #     gt_bboxes (Array[M, 4]| Array[N, 5]): Ground truth bounding boxes with xyxy/xyxyr format.
        #     gt_cls (Array[M]): The class labels.
        # """
        detections = torch.cat([pred_boxes_tensor, pred_conf_tensor.unsqueeze(1), pred_cls_tensor.unsqueeze(1)], dim=1)
        gt_bboxes = target_boxes_tensor
        gt_cls = target_cls_tensor
        self.confusion_matrix.process_batch(
            detections=detections,
            gt_bboxes=gt_bboxes,
            gt_cls=gt_cls,
        )
    
    # def update_metrics(
    #     self,
    #     prediction: dict[str, torch.Tensor],
    #     target: dict[str, torch.Tensor],
    #     *,
    #     image_name: str,
    # ) -> None:
    #     """Update DetMetrics and ConfusionMatrix for one image."""
    #     self.seen += 1
    #     target_cls_numpy = target["cls"].cpu().numpy()
    #     no_pred = prediction["cls"].numel() == 0

    #     self.metrics.update_stats(
    #         {
    #             **self._process_batch(prediction, target),
    #             "target_cls": target_cls_numpy,
    #             "target_img": np.unique(target_cls_numpy),
    #             "conf": (
    #                 np.zeros(0, dtype=np.float32)
    #                 if no_pred
    #                 else prediction["conf"].cpu().numpy()
    #             ),
    #             "pred_cls": (
    #                 np.zeros(0, dtype=np.float32)
    #                 if no_pred
    #                 else prediction["cls"].cpu().numpy()
    #             ),
    #             "im_name": image_name,
    #         }
    #     )

    #     self.confusion_matrix.process_batch(
    #         prediction,
    #         target,
    #         # class labels
    #         target_cls=target_cls_numpy,
    #     )
    
    def _process_batch(
        self,
        pred_boxes: torch.Tensor,
        pred_conf: torch.Tensor,
        pred_cls: torch.Tensor,
        target_boxes: torch.Tensor,
        target_cls: torch.Tensor,
    ) -> np.ndarray:
        """Create the [num_predictions, 10] TP matrix used by DetMetrics."""
        num_predictions = int(pred_cls.shape[0])
        if target_cls.numel() == 0 or num_predictions == 0:
            return np.zeros((num_predictions, self.niou), dtype=bool)

        iou = box_iou(target_boxes, pred_boxes)
        correct = self.match_predictions(pred_cls, target_cls, iou)
        return correct.cpu().numpy()
    
    # def _process_batch(
    #     self,
    #     prediction: dict[str, torch.Tensor],
    #     target: dict[str, torch.Tensor],
    # ) -> dict[str, np.ndarray]:
    #     """Create the [num_predictions, 10] TP matrix used by DetMetrics."""
    #     num_predictions = int(prediction["cls"].shape[0])
    #     if target["cls"].numel() == 0 or num_predictions == 0:
    #         return {
    #             "tp": np.zeros((num_predictions, self.niou), dtype=bool)
    #         }

    #     iou = box_iou(target["bboxes"], prediction["bboxes"])
    #     correct = self.match_predictions(
    #         prediction["cls"], target["cls"], iou
    #     )
    #     return {"tp": correct.cpu().numpy()}
    
    # def get_stats(self) -> dict[str, float]:
    #     """Process accumulated statistics and return Ultralytics result keys."""
    #     self.metrics.process(
    #         save_dir=self.save_dir,
    #         plot=bool(self.args.plots),
    #         on_plot=self.on_plot,
    #     )
    #     return self.metrics.results_dict
    
    def finalize_metrics(self) -> None:
        """Attach plots and confusion matrix to the metrics object."""
        if self.args.plots:
            for normalize in (True, False):
                self.confusion_matrix.plot(
                    save_dir=self.save_dir,
                    normalize=normalize,
                    on_plot=self.on_plot,
                )
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir
        
    # def print_results(self) -> None:
    #     """Print overall and per-class metrics in Ultralytics' style."""
    #     if self.metrics.nt_per_class is None:
    #         return

    #     pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)
    #     total_instances = int(self.metrics.nt_per_class.sum())
    #     LOGGER.info(
    #         pf
    #         % (
    #             "all",
    #             self.seen,
    #             total_instances,
    #             *self.metrics.mean_results(),
    #         )
    #     )

    #     if self.args.verbose and self.nc > 1:
    #         for metric_index, class_id in enumerate(self.metrics.ap_class_index):
    #             class_id = int(class_id)
    #             LOGGER.info(
    #                 pf
    #                 % (
    #                     self.names[class_id],
    #                     int(self.metrics.nt_per_image[class_id]),
    #                     int(self.metrics.nt_per_class[class_id]),
    #                     *self.metrics.class_result(metric_index),
    #                 )
    #             )
    
    def _prepare_ground_truth(
        self,
        record: AnnotationRecord,
        *,
        box_format: BoxFormat,
        image_width: int,
        image_height: int,
        label_offset: int,
        image_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        boxes = self._convert_boxes_to_xyxyn(
            record.get("bboxes", []),
            box_format=box_format,
            image_width=image_width,
            image_height=image_height,
            source=f"ground truth for {image_name}",
        )
        classes = self._prepare_classes(
            record.get("cls", record.get("labels", [])),
            label_offset=label_offset,
            expected_length=boxes.shape[0],
            source=f"ground truth for {image_name}",
        )
        if self.args.single_cls:
            classes.zero_()
        return boxes, classes
    
    def _prepare_prediction(
        self,
        record: AnnotationRecord,
        *,
        box_format: BoxFormat,
        image_width: int,
        image_height: int,
        label_offset: int,
        image_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes = self._convert_boxes_to_xyxyn(
            record.get("bboxes", []),
            box_format=box_format,
            image_width=image_width,
            image_height=image_height,
            source=f"predictions for {image_name}",
        )
        classes = self._prepare_classes(
            record.get("cls", record.get("labels", [])),
            label_offset=label_offset,
            expected_length=boxes.shape[0],
            source=f"predictions for {image_name}",
        )
        confidence = torch.as_tensor(
            record.get("conf", record.get("scores", [])),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)

        if confidence.shape[0] != boxes.shape[0]:
            raise ValueError(
                f"predictions for {image_name}: found {boxes.shape[0]} boxes "
                f"but {confidence.shape[0]} confidence values"
            )
        if not torch.isfinite(confidence).all():
            raise ValueError(
                f"predictions for {image_name}: confidence contains NaN or infinity"
            )
        if confidence.numel() and (
            confidence.min() < 0 or confidence.max() > 1
        ):
            raise ValueError(
                f"predictions for {image_name}: confidence must be in [0, 1]"
            )

        if self.args.single_cls:
            classes.zero_()

        keep = confidence >= self.min_conf
        boxes, classes, confidence = boxes[keep], classes[keep], confidence[keep]

        if confidence.numel():
            order = torch.argsort(confidence, descending=True)[: self.max_det]
            boxes, classes, confidence = (
                boxes[order],
                classes[order],
                confidence[order],
            )

        return boxes, classes, confidence
    
    def _convert_boxes_to_xyxyn(
        self,
        boxes: Any,
        *,
        box_format: BoxFormat,
        image_width: int,
        image_height: int,
        source: str,
    ) -> torch.Tensor:
        """Convert supported box formats to normalized XYXY."""
        tensor = torch.as_tensor(
            boxes, dtype=torch.float32, device=self.device
        )
        if tensor.numel() == 0:
            return torch.empty((0, 4), dtype=torch.float32, device=self.device)

        try:
            tensor = tensor.reshape(-1, 4)
        except RuntimeError as exc:
            raise ValueError(f"{source}: boxes must have shape [N, 4]") from exc

        if not torch.isfinite(tensor).all():
            raise ValueError(f"{source}: boxes contain NaN or infinity")

        if box_format in {"xywh", "xywhn"}:
            tensor = ops.xywh2xyxy(tensor)
        elif box_format not in {"xyxy", "xyxyn"}:
            raise ValueError(f"Unsupported box format {box_format!r}")

        if box_format in {"xyxy", "xywh"}:
            tensor = tensor / torch.tensor(
                [image_width, image_height, image_width, image_height],
                dtype=tensor.dtype,
                device=self.device,
            )

        tolerance = 1e-4
        if tensor.min() < -tolerance or tensor.max() > 1 + tolerance:
            raise ValueError(
                f"{source}: normalized coordinates fall outside [0, 1]. "
                "Check the declared box format."
            )

        tensor = tensor.clamp(0.0, 1.0)
        if ((tensor[:, 2] - tensor[:, 0]) <= 0).any() or (
            (tensor[:, 3] - tensor[:, 1]) <= 0
        ).any():
            raise ValueError(
                f"{source}: every box must have positive width and height"
            )
        return tensor
    
    def _prepare_classes(
        self,
        classes: Any,
        *,
        label_offset: int,
        expected_length: int,
        source: str,
    ) -> torch.Tensor:
        values = torch.as_tensor(classes, device=self.device).reshape(-1)
        if values.shape[0] != expected_length:
            raise ValueError(
                f"{source}: found {expected_length} boxes but "
                f"{values.shape[0]} class labels"
            )
        if values.numel() == 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        values = values.float()
        if not torch.isfinite(values).all() or not torch.equal(values, values.round()):
            raise ValueError(f"{source}: class labels must be finite integers")

        values = values - int(label_offset)
        if values.min() < 0 or values.max() >= self.nc:
            raise ValueError(
                f"{source}: class IDs must resolve to [0, {self.nc - 1}] "
                f"after subtracting label_offset={label_offset}"
            )
        return values.float()
    
    @staticmethod
    def _normalize_names(
        names: Sequence[str] | Mapping[int, str],
    ) -> dict[int, str]:
        normalized = (
            {int(class_id): str(name) for class_id, name in names.items()}
            if isinstance(names, Mapping)
            else {class_id: str(name) for class_id, name in enumerate(names)}
        )
        if not normalized:
            raise ValueError("At least one class name is required")
        if sorted(normalized) != list(range(len(normalized))):
            raise ValueError("names must use consecutive zero-based class IDs")
        return normalized
        
    @staticmethod
    def _validate_collection(
        records: RecordCollection,
        number_of_images: int,
        collection_name: str,
    ) -> None:
        if not isinstance(records, Mapping) and len(records) != number_of_images:
            raise ValueError(
                f"{collection_name} has {len(records)} records, but "
                f"{number_of_images} images were supplied"
            )
            
    @staticmethod
    def _resolve_record(
        records: RecordCollection,
        image_path: Path,
        image_index: int,
        collection_name: str,
    ) -> AnnotationRecord:
        if not isinstance(records, Mapping):
            return records[image_index]

        for key in (
            image_path,
            str(image_path),
            image_path.as_posix(),
            image_path.name,
            image_path.stem,
        ):
            if key in records:
                return records[key]

        raise KeyError(
            f"No {collection_name} record found for {image_path}. Use the "
            "full path, filename, or stem as the mapping key."
        )
    
    @staticmethod
    def predictions_from_lists(
        images: Sequence[str | Path],
        boxes: Sequence[Any],
        scores: Sequence[Any],
        labels: Sequence[Any],
    ) -> dict[str, dict[str, Any]]:
        """Convert Predictor/EnsemblePredictor outputs into records."""
        if len({len(images), len(boxes), len(scores), len(labels)}) != 1:
            raise ValueError(
                "images, boxes, scores, and labels must have equal lengths"
            )
        return {
            Path(image_path).name: {
                "bboxes": image_boxes,
                "conf": image_scores,
                "cls": image_labels,
            }
            for image_path, image_boxes, image_scores, image_labels in zip(
                images, boxes, scores, labels
            )
        }
    
    @staticmethod
    def load_ground_truths(
        images: Sequence[str | Path],
        labels_dir: str | Path | None = None,
        *,
        allow_missing_files: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """
        Load YOLO rows: class_id center_x center_y width height.

        ORDDC-style class tokens such as `3:low` are accepted; only the numeric
        class ID before ':' is used for detection evaluation.
        """
        labels_dir = Path(labels_dir) if labels_dir is not None else None
        ground_truths: dict[str, dict[str, Any]] = {}

        for image_value in images:
            image_path = Path(image_value)
            # label_path = labels_dir / f"{image_path.stem}.txt"
            image_dir = image_path.parent
            label_placeholder_path = image_dir.parent / "labels" / f"{image_path.stem}.txt"
            label_path = labels_dir / f"{image_path.stem}.txt" if labels_dir is not None else label_placeholder_path
            
            classes: list[int] = []
            boxes: list[list[float]] = []

            if label_path.is_file():
                with label_path.open("r", encoding="utf-8") as file:
                    for line_number, raw_line in enumerate(file, start=1):
                        parts = raw_line.strip().split()
                        if not parts:
                            continue
                        if len(parts) < 5:
                            raise ValueError(
                                f"{label_path}:{line_number}: expected at least "
                                "five values"
                            )
                        try:
                            class_id = int(float(parts[0].split(":", 1)[0]))
                            box = [float(value) for value in parts[1:5]]
                        except ValueError as exc:
                            raise ValueError(
                                f"{label_path}:{line_number}: invalid YOLO annotation"
                            ) from exc
                        classes.append(class_id)
                        boxes.append(box)
            elif not allow_missing_files:
                raise FileNotFoundError(f"Label file not found: {label_path}")

            ground_truths[image_path.name] = {
                "bboxes": boxes,
                "cls": classes,
            }

        return ground_truths
