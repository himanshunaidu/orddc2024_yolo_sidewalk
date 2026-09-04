from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou

from ..predictions.prediction_result import PredictionResult
from .validation_result import ValidationResult


BoxFormat = Literal["xyxy", "xyxyn", "xywh", "xywhn"]
AnnotationRecord = Mapping[str, Any]
RecordCollection = (
    Sequence[AnnotationRecord]
    | Mapping[str | Path, AnnotationRecord]
)


class CustomValidator(BaseValidator):
    """
    Evaluate already-postprocessed object-detection predictions.

    PredictionResult contract:
        - boxes are normalized XYXY
        - labels are zero-based
        - confidence scores are in [0, 1]

    Ground-truth record format:
        {
            "bboxes": array-like [N, 4],
            "cls": array-like [N],
        }

    Ground-truth records may be:
        - a sequence aligned with PredictionResult.images, or
        - a mapping keyed by full path, filename, or stem.

    When plots=True, Ultralytics DetMetrics/ap_per_class generates:
        - PR_curve.png
        - F1_curve.png
        - P_curve.png
        - R_curve.png

    Confusion-matrix plots are generated separately in finalize_metrics().
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

        self.iouv = torch.linspace(
            0.50,
            0.95,
            10,
            device=self.device,
        )
        self.niou = int(self.iouv.numel())

        self.min_conf = float(min_conf)
        self.max_det = int(max_det)

        self.confusion_conf = float(confusion_conf)
        self.confusion_iou = float(confusion_iou)

        self.reset()

    def evaluate(
        self,
        predictions: PredictionResult,
        ground_truths: RecordCollection,
        *,
        ground_truth_box_format: BoxFormat = "xywhn",
        ground_truth_label_offset: int = 0,
    ) -> ValidationResult:
        """
        Evaluate one PredictionResult against ground truth.
        """
        image_paths = [
            Path(path)
            for path in predictions.images
        ]

        if not image_paths:
            raise ValueError(
                "PredictionResult must contain at least one image"
            )

        self._validate_collection(
            ground_truths,
            len(image_paths),
            "ground_truths",
        )

        self.reset()

        total_targets = 0

        tp_parts: list[np.ndarray] = []
        conf_parts: list[np.ndarray] = []
        pred_cls_parts: list[np.ndarray] = []
        target_cls_parts: list[np.ndarray] = []

        for image_index, image_path in enumerate(image_paths):
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            with Image.open(image_path) as image:
                image_width, image_height = image.size

            gt_record = self._resolve_record(
                ground_truths,
                image_path,
                image_index,
                "ground_truths",
            )

            target_boxes, target_cls = self._prepare_ground_truth(
                gt_record,
                box_format=ground_truth_box_format,
                image_width=image_width,
                image_height=image_height,
                label_offset=ground_truth_label_offset,
                image_name=image_path.name,
            )

            pred_boxes, pred_cls, pred_conf = self._prepare_prediction_result(
                predictions,
                image_index=image_index,
                image_name=image_path.name,
            )

            tp = self._process_batch(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target_boxes=target_boxes,
                target_cls=target_cls,
            )

            total_targets += int(target_cls.numel())

            tp_parts.append(tp)
            conf_parts.append(
                pred_conf.detach().cpu().numpy()
            )
            pred_cls_parts.append(
                pred_cls.detach().cpu().numpy()
            )
            target_cls_parts.append(
                target_cls.detach().cpu().numpy()
            )

            # Confusion-matrix matching must stay image-local.
            self._update_confusion_matrix(
                pred_boxes=pred_boxes,
                pred_conf=pred_conf,
                pred_cls=pred_cls,
                target_boxes=target_boxes,
                target_cls=target_cls,
            )

        if total_targets == 0:
            raise ValueError(
                "No ground-truth objects were found. Detection AP cannot be "
                "calculated without target annotations."
            )

        tp = np.concatenate(
            tp_parts,
            axis=0,
        )
        conf = np.concatenate(
            conf_parts,
            axis=0,
        )
        pred_cls = np.concatenate(
            pred_cls_parts,
            axis=0,
        )
        target_cls = np.concatenate(
            target_cls_parts,
            axis=0,
        )

        # For the Ultralytics 8.2.x DetMetrics implementation used by this
        # project, plot/save_dir/on_plot are configured on the DetMetrics
        # instance in reset(). Calling process() here therefore computes both
        # the metrics and the standard PR/F1/P/R curves.
        self.metrics.process(
            tp=tp,
            conf=conf,
            pred_cls=pred_cls,
            target_cls=target_cls,
        )

        per_class = self._extract_per_class_metrics(
            target_cls=target_cls,
        )
        
        average_detections_per_image = self._average_detections_per_image(tp)

        self.finalize_metrics()

        return ValidationResult(
            overall=dict(self.metrics.results_dict),
            confusion_matrix=self.confusion_matrix.matrix.copy(),
            metrics=self.metrics,
            per_class=per_class,
            average_detections_per_image=average_detections_per_image,
        )

    def evaluate_predictor(
        self,
        predictor: Any,
        ground_truths: RecordCollection,
        *,
        model_index: int = 0,
        ground_truth_box_format: BoxFormat = "xywhn",
        ground_truth_label_offset: int = 0,
    ) -> ValidationResult:
        prediction_results = predictor.predict()

        if not prediction_results:
            raise ValueError(
                "Predictor returned no PredictionResult objects"
            )

        if not 0 <= model_index < len(prediction_results):
            raise IndexError(
                f"model_index={model_index} is out of range for "
                f"{len(prediction_results)} prediction result(s)"
            )

        return self.evaluate(
            predictions=prediction_results[model_index],
            ground_truths=ground_truths,
            ground_truth_box_format=ground_truth_box_format,
            ground_truth_label_offset=ground_truth_label_offset,
        )

    def evaluate_cache(
        self,
        prediction_cache: str | Path,
        ground_truths: RecordCollection,
        *,
        ground_truth_box_format: BoxFormat = "xywhn",
        ground_truth_label_offset: int = 0,
    ) -> ValidationResult:
        predictions = PredictionResult.load_npz(
            prediction_cache
        )

        return self.evaluate(
            predictions=predictions,
            ground_truths=ground_truths,
            ground_truth_box_format=ground_truth_box_format,
            ground_truth_label_offset=ground_truth_label_offset,
        )

    def reset(self) -> None:
        """Reset evaluator state before a validation run."""
        self.seen = 0

        # Ultralytics 8.2.x DetMetrics owns the AP-curve plotting options.
        # With plot=True, DetMetrics.process() calls ap_per_class(), which
        # writes PR_curve.png, F1_curve.png, P_curve.png, and R_curve.png.
        self.metrics = DetMetrics(
            save_dir=self.save_dir,
            plot=bool(self.args.plots),
            on_plot=self.on_plot,
            names=self.names,
        )
        self.metrics.names = self.names

        if hasattr(
            self.metrics,
            "clear_image_metrics",
        ):
            self.metrics.clear_image_metrics()

        self.confusion_matrix = ConfusionMatrix(
            nc=self.nc,
            task="detect",
            conf=self.confusion_conf,
            iou_thres=self.confusion_iou,
        )

        self.plots = {}

    def _extract_per_class_metrics(
        self,
        *,
        target_cls: np.ndarray,
    ) -> dict[int, dict[str, Any]]:
        """
        Return Ultralytics class-wise P/R/F1/AP metrics.

        Precision and recall use the same global max-mean-F1 operating point
        chosen internally by Ultralytics ap_per_class(). AP50 and AP50-95 use
        the complete confidence-ranked precision-recall curves.

        DetMetrics.class_result(i) indexes metric rows, while
        DetMetrics.ap_class_index maps those rows back to actual class IDs.
        """
        target_ids = np.asarray(
            target_cls,
            dtype=np.int64,
        ).reshape(-1)

        target_counts = np.bincount(
            target_ids,
            minlength=self.nc,
        )

        per_class: dict[int, dict[str, Any]] = {
            class_id: {
                "class_id": class_id,
                "class_name": self.names[class_id],
                "targets": int(target_counts[class_id]),
                "precision": None,
                "recall": None,
                "f1": None,
                "mAP50": None,
                "mAP50-95": None,
            }
            for class_id in range(self.nc)
        }

        ap_class_index = np.asarray(
            self.metrics.ap_class_index,
            dtype=np.int64,
        ).reshape(-1)

        for metric_index, class_id_value in enumerate(
            ap_class_index
        ):
            class_id = int(class_id_value)

            precision, recall, ap50, ap50_95 = (
                self.metrics.class_result(
                    metric_index
                )
            )

            precision = float(precision)
            recall = float(recall)
            ap50 = float(ap50)
            ap50_95 = float(ap50_95)

            denominator = precision + recall
            f1 = (
                2.0 * precision * recall / denominator
                if denominator > 0.0
                else 0.0
            )

            per_class[class_id].update(
                {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mAP50": ap50,
                    "mAP50-95": ap50_95,
                }
            )

        return per_class
    
    def _average_detections_per_image(
        self,
        correct: np.ndarray,
    ) -> float:
        """
        Compute the average number of correct detections per image.
        """
        if correct.size == 0:
            return 0.0
        return float(correct.sum() / correct.shape[0])

    def _process_batch(
        self,
        pred_boxes: torch.Tensor,
        pred_cls: torch.Tensor,
        target_boxes: torch.Tensor,
        target_cls: torch.Tensor,
    ) -> np.ndarray:
        num_predictions = int(
            pred_cls.shape[0]
        )

        if (
            target_cls.numel() == 0
            or num_predictions == 0
        ):
            return np.zeros(
                (
                    num_predictions,
                    self.niou,
                ),
                dtype=bool,
            )

        iou = box_iou(
            target_boxes,
            pred_boxes,
        )

        correct = self.match_predictions(
            pred_cls,
            target_cls,
            iou,
        )

        return correct.detach().cpu().numpy()

    def _update_confusion_matrix(
        self,
        *,
        pred_boxes: torch.Tensor,
        pred_conf: torch.Tensor,
        pred_cls: torch.Tensor,
        target_boxes: torch.Tensor,
        target_cls: torch.Tensor,
    ) -> None:
        detections = torch.cat(
            [
                pred_boxes,
                pred_conf.unsqueeze(1),
                pred_cls.unsqueeze(1),
            ],
            dim=1,
        )

        self.confusion_matrix.process_batch(
            detections=detections,
            gt_bboxes=target_boxes,
            gt_cls=target_cls,
        )

    def finalize_metrics(self) -> None:
        """
        Plot confusion matrices and attach evaluator state to DetMetrics.

        PR/F1/P/R curves are already generated by DetMetrics.process() when
        plots=True.
        """
        if self.args.plots:
            for normalize in (
                True,
                False,
            ):
                self.confusion_matrix.plot(
                    save_dir=self.save_dir,
                    normalize=normalize,
                    on_plot=self.on_plot,
                )

        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir

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
            record.get(
                "bboxes",
                [],
            ),
            box_format=box_format,
            image_width=image_width,
            image_height=image_height,
            source=f"ground truth for {image_name}",
        )

        classes = self._prepare_classes(
            record.get(
                "cls",
                record.get(
                    "labels",
                    [],
                ),
            ),
            label_offset=label_offset,
            expected_length=boxes.shape[0],
            source=f"ground truth for {image_name}",
        )

        if self.args.single_cls:
            classes.zero_()

        return boxes, classes

    def _prepare_prediction_result(
        self,
        predictions: PredictionResult,
        *,
        image_index: int,
        image_name: str,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        boxes = self._convert_boxes_to_xyxyn(
            predictions.boxes[image_index],
            box_format="xyxyn",
            image_width=1,
            image_height=1,
            source=f"predictions for {image_name}",
        )

        classes = self._prepare_classes(
            predictions.labels[image_index],
            label_offset=0,
            expected_length=boxes.shape[0],
            source=f"predictions for {image_name}",
        )

        confidence = torch.as_tensor(
            predictions.scores[image_index],
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)

        if confidence.shape[0] != boxes.shape[0]:
            raise ValueError(
                f"predictions for {image_name}: found "
                f"{boxes.shape[0]} boxes but "
                f"{confidence.shape[0]} confidence values"
            )

        if not torch.isfinite(
            confidence
        ).all():
            raise ValueError(
                f"predictions for {image_name}: confidence contains "
                "NaN or infinity"
            )

        if confidence.numel() and (
            confidence.min() < 0
            or confidence.max() > 1
        ):
            raise ValueError(
                f"predictions for {image_name}: confidence must be in [0, 1]"
            )

        if self.args.single_cls:
            classes.zero_()

        keep = (
            confidence
            >= self.min_conf
        )

        boxes = boxes[keep]
        classes = classes[keep]
        confidence = confidence[keep]

        if confidence.numel():
            order = torch.argsort(
                confidence,
                descending=True,
            )[: self.max_det]

            boxes = boxes[order]
            classes = classes[order]
            confidence = confidence[order]

        return (
            boxes,
            classes,
            confidence,
        )

    def _convert_boxes_to_xyxyn(
        self,
        boxes: Any,
        *,
        box_format: BoxFormat,
        image_width: int,
        image_height: int,
        source: str,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(
            boxes,
            dtype=torch.float32,
            device=self.device,
        )

        if tensor.numel() == 0:
            return torch.empty(
                (0, 4),
                dtype=torch.float32,
                device=self.device,
            )

        try:
            tensor = tensor.reshape(
                -1,
                4,
            )
        except RuntimeError as exc:
            raise ValueError(
                f"{source}: boxes must have shape [N, 4]"
            ) from exc

        if not torch.isfinite(
            tensor
        ).all():
            raise ValueError(
                f"{source}: boxes contain NaN or infinity"
            )

        if box_format in {
            "xywh",
            "xywhn",
        }:
            tensor = ops.xywh2xyxy(
                tensor
            )
        elif box_format not in {
            "xyxy",
            "xyxyn",
        }:
            raise ValueError(
                f"Unsupported box format {box_format!r}"
            )

        if box_format in {
            "xyxy",
            "xywh",
        }:
            tensor = tensor / torch.tensor(
                [
                    image_width,
                    image_height,
                    image_width,
                    image_height,
                ],
                dtype=tensor.dtype,
                device=self.device,
            )

        tolerance = 1e-4

        if (
            tensor.min() < -tolerance
            or tensor.max() > 1 + tolerance
        ):
            raise ValueError(
                f"{source}: normalized coordinates fall outside [0, 1]. "
                "Check the declared box format."
            )

        tensor = tensor.clamp(
            0.0,
            1.0,
        )

        if (
            (
                tensor[:, 2]
                - tensor[:, 0]
            )
            < 0
        ).any() or (
            (
                tensor[:, 3]
                - tensor[:, 1]
            )
            < 0
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
        values = torch.as_tensor(
            classes,
            device=self.device,
        ).reshape(-1)

        if values.shape[0] != expected_length:
            raise ValueError(
                f"{source}: found {expected_length} boxes but "
                f"{values.shape[0]} class labels"
            )

        if values.numel() == 0:
            return torch.empty(
                0,
                dtype=torch.float32,
                device=self.device,
            )

        values = values.float()

        if (
            not torch.isfinite(
                values
            ).all()
            or not torch.equal(
                values,
                values.round(),
            )
        ):
            raise ValueError(
                f"{source}: class labels must be finite integers"
            )

        values = (
            values
            - int(label_offset)
        )

        if (
            values.min() < 0
            or values.max() >= self.nc
        ):
            raise ValueError(
                f"{source}: class IDs must resolve to "
                f"[0, {self.nc - 1}] after subtracting "
                f"label_offset={label_offset}"
            )

        return values.float()

    @staticmethod
    def _normalize_names(
        names: Sequence[str]
        | Mapping[int, str],
    ) -> dict[int, str]:
        normalized = (
            {
                int(class_id): str(name)
                for class_id, name in names.items()
            }
            if isinstance(
                names,
                Mapping,
            )
            else {
                class_id: str(name)
                for class_id, name in enumerate(
                    names
                )
            }
        )

        if not normalized:
            raise ValueError(
                "At least one class name is required"
            )

        if sorted(
            normalized
        ) != list(
            range(
                len(normalized)
            )
        ):
            raise ValueError(
                "names must use consecutive zero-based class IDs"
            )

        return normalized

    @staticmethod
    def _validate_collection(
        records: RecordCollection,
        number_of_images: int,
        collection_name: str,
    ) -> None:
        if (
            not isinstance(
                records,
                Mapping,
            )
            and len(records)
            != number_of_images
        ):
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
        if not isinstance(
            records,
            Mapping,
        ):
            return records[
                image_index
            ]

        for key in (
            image_path,
            str(image_path),
            image_path.as_posix(),
            image_path.name,
            image_path.stem,
        ):
            if key in records:
                return records[
                    key
                ]

        raise KeyError(
            f"No {collection_name} record found for {image_path}. Use the "
            "full path, filename, or stem as the mapping key."
        )

    @staticmethod
    def load_ground_truths(
        images: Sequence[str | Path],
        labels_dir: str | Path | None = None,
        *,
        allow_missing_files: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """
        Load YOLO rows:

            class_id center_x center_y width height

        ORDDC-style class tokens such as `3:low` are accepted; only the
        numeric class ID before ':' is used for detection evaluation.
        """
        labels_dir = (
            Path(labels_dir)
            if labels_dir is not None
            else None
        )

        ground_truths: dict[
            str,
            dict[str, Any],
        ] = {}

        for image_value in images:
            image_path = Path(
                image_value
            )

            image_dir = image_path.parent

            inferred_label_path = (
                image_dir.parent
                / "labels"
                / f"{image_path.stem}.txt"
            )

            label_path = (
                labels_dir
                / f"{image_path.stem}.txt"
                if labels_dir is not None
                else inferred_label_path
            )

            classes: list[int] = []
            boxes: list[list[float]] = []

            if label_path.is_file():
                with label_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    for line_number, raw_line in enumerate(
                        file,
                        start=1,
                    ):
                        parts = raw_line.strip().split()

                        if not parts:
                            continue

                        if len(parts) < 5:
                            raise ValueError(
                                f"{label_path}:{line_number}: expected "
                                "at least five values"
                            )

                        try:
                            class_id = int(
                                float(
                                    parts[0].split(
                                        ":",
                                        1,
                                    )[0]
                                )
                            )

                            box = [
                                float(value)
                                for value
                                in parts[1:5]
                            ]

                        except ValueError as exc:
                            raise ValueError(
                                f"{label_path}:{line_number}: invalid "
                                "YOLO annotation"
                            ) from exc

                        classes.append(
                            class_id
                        )
                        boxes.append(
                            box
                        )

            elif not allow_missing_files:
                raise FileNotFoundError(
                    f"Label file not found: {label_path}"
                )

            ground_truths[
                str(image_path)
            ] = {
                "bboxes": boxes,
                "cls": classes,
            }

        return ground_truths