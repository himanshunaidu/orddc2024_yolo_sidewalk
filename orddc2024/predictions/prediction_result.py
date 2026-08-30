from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PREDICTION_FORMAT_VERSION = 1


@dataclass(slots=True)
class PredictionResult:
    """
    Canonical predictions for one logical model output.

    Contract:
      - boxes: normalized XYXY ([x1, y1, x2, y2] in [0, 1])
      - labels: zero-based integer class IDs
      - scores: confidence values in [0, 1]
    """

    images: list[str]
    boxes: list[list[list[float]]]
    scores: list[list[float]]
    labels: list[list[int]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.images = [str(x) for x in self.images]
        self.boxes = [
            [[float(v) for v in box] for box in image_boxes]
            for image_boxes in self.boxes
        ]
        self.scores = [
            [float(v) for v in image_scores]
            for image_scores in self.scores
        ]
        self.labels = [
            [int(v) for v in image_labels]
            for image_labels in self.labels
        ]
        self.metadata = dict(self.metadata)
        self._validate()

    @property
    def box_format(self) -> str:
        return "xyxyn"

    @property
    def label_offset(self) -> int:
        return 0

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_detections(self) -> int:
        return sum(len(x) for x in self.boxes)

    def _validate(self) -> None:
        if not (
            len(self.images)
            == len(self.boxes)
            == len(self.scores)
            == len(self.labels)
        ):
            raise ValueError(
                "images, boxes, scores, and labels must have equal image counts"
            )

        if len(set(self.images)) != len(self.images):
            raise ValueError("PredictionResult.images contains duplicates")

        for image_name, image_boxes, image_scores, image_labels in zip(
            self.images, self.boxes, self.scores, self.labels
        ):
            if not (
                len(image_boxes)
                == len(image_scores)
                == len(image_labels)
            ):
                raise ValueError(
                    f"{image_name}: boxes, scores, and labels must have "
                    "equal detection counts"
                )

            for box in image_boxes:
                if len(box) != 4:
                    raise ValueError(f"{image_name}: each box must have 4 values")
                values = np.asarray(box, dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError(f"{image_name}: box contains NaN/inf")
                if values.min() < -1e-6 or values.max() > 1.0 + 1e-6:
                    raise ValueError(
                        f"{image_name}: xyxyn coordinates must be in [0, 1]"
                    )
                x1, y1, x2, y2 = values
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(
                        f"{image_name}: box must have positive width/height"
                    )

            for score in image_scores:
                if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError(
                        f"{image_name}: confidence must be in [0, 1]"
                    )

            for label in image_labels:
                if label < 0:
                    raise ValueError(
                        f"{image_name}: class IDs must be zero-based "
                        "non-negative integers"
                    )

    def save_npz(
        self,
        path: str | Path,
        *,
        compressed: bool = True,
    ) -> Path:
        """
        Save as a compact non-object NumPy archive.

        Ragged per-image detections are flattened and reconstructed with
        image_offsets.
        """
        path = Path(path).expanduser()
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)

        offsets = np.zeros(self.num_images + 1, dtype=np.int64)
        for i, image_boxes in enumerate(self.boxes):
            offsets[i + 1] = offsets[i] + len(image_boxes)

        flat_boxes = (
            np.asarray(
                [b for image_boxes in self.boxes for b in image_boxes],
                dtype=np.float32,
            ).reshape(-1, 4)
            if self.num_detections
            else np.empty((0, 4), dtype=np.float32)
        )
        flat_scores = np.asarray(
            [s for image_scores in self.scores for s in image_scores],
            dtype=np.float32,
        )
        flat_labels = np.asarray(
            [l for image_labels in self.labels for l in image_labels],
            dtype=np.int64,
        )

        payload = {
            "format_version": np.asarray(
                [PREDICTION_FORMAT_VERSION], dtype=np.int64
            ),
            "box_format": np.asarray(["xyxyn"]),
            "label_offset": np.asarray([0], dtype=np.int64),
            "images": np.asarray(self.images, dtype=np.str_),
            "image_offsets": offsets,
            "boxes": flat_boxes,
            "scores": flat_scores,
            "labels": flat_labels,
            "metadata_json": np.asarray(
                [json.dumps(self._json_safe(self.metadata))]
            ),
        }

        if compressed:
            np.savez_compressed(path, **payload)
        else:
            np.savez(path, **payload)

        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> "PredictionResult":
        path = Path(path).expanduser()

        with np.load(path, allow_pickle=False) as archive:
            version = int(archive["format_version"][0])
            if version != PREDICTION_FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported PredictionResult format version {version}"
                )

            if str(archive["box_format"][0]) != "xyxyn":
                raise ValueError("Prediction cache must use xyxyn boxes")
            if int(archive["label_offset"][0]) != 0:
                raise ValueError("Prediction cache must use zero-based labels")

            images = archive["images"].astype(str).tolist()
            offsets = archive["image_offsets"].astype(np.int64)
            flat_boxes = archive["boxes"].astype(np.float32)
            flat_scores = archive["scores"].astype(np.float32)
            flat_labels = archive["labels"].astype(np.int64)
            metadata = json.loads(str(archive["metadata_json"][0]))

        if len(offsets) != len(images) + 1:
            raise ValueError("Invalid image_offsets length")
        if offsets[0] != 0:
            raise ValueError("image_offsets must start at zero")
        if offsets[-1] != len(flat_boxes):
            raise ValueError("Final image offset does not match detections")
        if not (
            len(flat_boxes) == len(flat_scores) == len(flat_labels)
        ):
            raise ValueError(
                "Flattened boxes/scores/labels have different lengths"
            )

        boxes, scores, labels = [], [], []
        for i in range(len(images)):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            boxes.append(flat_boxes[start:end].tolist())
            scores.append(flat_scores[start:end].tolist())
            labels.append(flat_labels[start:end].tolist())

        return cls(
            images=images,
            boxes=boxes,
            scores=scores,
            labels=labels,
            metadata=metadata,
        )

    def save_orddc_folder(
        self,
        output_dir: str | Path,
        *,
        dataset_root: str | Path,
        include_scores: bool = True,
        write_manifest: bool = True,
    ) -> Path:
        """
        Save one TXT file per sample, mirroring its dataset-relative path.

        Example:
            <dataset_root>/val/images/a.jpg
        becomes:
            <output_dir>/val/images/a.txt

        Row format with scores:
            class_id x1 y1 x2 y2 confidence

        Row format without scores:
            class_id x1 y1 x2 y2

        Boxes are normalized XYXY and classes are zero-based.
        """
        output_dir = Path(output_dir).expanduser().resolve()
        dataset_root = Path(dataset_root).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest = []

        for image, image_boxes, image_scores, image_labels in zip(
            self.images, self.boxes, self.scores, self.labels
        ):
            image_path = Path(image).expanduser()
            if not image_path.is_absolute():
                image_path = dataset_root / image_path
            image_path = image_path.resolve()

            try:
                relative_image = image_path.relative_to(dataset_root)
            except ValueError as exc:
                raise ValueError(
                    f"Image {image_path} is not under dataset_root "
                    f"{dataset_root}"
                ) from exc

            relative_txt = relative_image.with_suffix(".txt")
            output_path = output_dir / relative_txt
            output_path.parent.mkdir(parents=True, exist_ok=True)

            lines = []
            for box, score, label in zip(
                image_boxes, image_scores, image_labels
            ):
                x1, y1, x2, y2 = box
                if include_scores:
                    lines.append(
                        f"{label} "
                        f"{x1:.8f} {y1:.8f} {x2:.8f} {y2:.8f} "
                        f"{score:.8f}"
                    )
                else:
                    lines.append(
                        f"{label} "
                        f"{x1:.8f} {y1:.8f} {x2:.8f} {y2:.8f}"
                    )

            output_path.write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )

            manifest.append(
                {
                    "image": relative_image.as_posix(),
                    "predictions": relative_txt.as_posix(),
                    "num_detections": len(image_boxes),
                }
            )

        if write_manifest:
            payload = {
                "format_version": PREDICTION_FORMAT_VERSION,
                "box_format": "xyxyn",
                "label_offset": 0,
                "includes_scores": include_scores,
                "dataset_root": str(dataset_root),
                "metadata": self._json_safe(self.metadata),
                "samples": manifest,
            }
            (output_dir / "prediction_manifest.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )

        return output_dir

    def save(self, path: str | Path) -> Path:
        return self.save_npz(path)

    @classmethod
    def load(cls, path: str | Path) -> "PredictionResult":
        return cls.load_npz(path)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {
                str(k): PredictionResult._json_safe(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [PredictionResult._json_safe(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)