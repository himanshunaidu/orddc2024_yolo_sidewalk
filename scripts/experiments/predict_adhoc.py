from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import smooth

from orddc2024.evaluation.custom_validator import CustomValidator
from orddc2024.predictions.prediction_result import PredictionResult


WEIGHT = "/rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk/A_masked_2_full_orddc_weights_yolov8x_batch_16_lr0_0.001_lrf_0.01_imgsz_640_opt_SGD_lr1e-32/weights/best.pt"

DATASET_ROOT = Path("/rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk")

VAL_IMAGES = DATASET_ROOT / "val.txt"
DATA_YAML = DATASET_ROOT / "train.yaml"
LABELS_DIR = DATASET_ROOT / "labels"

PREDICTION_OUTPUT = Path(
    "runs/native/native_predict_recheck_predictions.npz"
)

VALIDATION_OUTPUT_DIR = Path(
    "runs/native/native_predict_recheck_custom_validation"
)


def extract_smoothed_f1_peak(
    validator: CustomValidator,
) -> tuple[float, float] | None:
    box_metrics = getattr(
        getattr(validator, "metrics", None),
        "box",
        None,
    )

    if box_metrics is None:
        return None

    px = getattr(box_metrics, "px", None)
    f1_curve = getattr(box_metrics, "f1_curve", None)

    if px is None or f1_curve is None:
        return None

    px = np.asarray(px, dtype=np.float64)
    f1_curve = np.asarray(f1_curve, dtype=np.float64)

    if (
        f1_curve.ndim != 2
        or f1_curve.shape[0] == 0
        or f1_curve.shape[1] != px.size
        or px.size == 0
    ):
        return None

    mean_f1 = f1_curve.mean(axis=0)
    smoothed_f1 = np.asarray(
        smooth(mean_f1, 0.05),
        dtype=np.float64,
    )

    best_index = int(np.argmax(smoothed_f1))

    return (
        float(smoothed_f1[best_index]),
        float(px[best_index]),
    )


# ----------------------------------------------------------------------
# Direct Ultralytics prediction
# ----------------------------------------------------------------------

model = YOLO(WEIGHT)

native_results = model.predict(
    source=str(VAL_IMAGES),
    imgsz=640,
    batch=32,
    device="0",
    conf=0.001,
    iou=0.7,
    agnostic_nms=False,
    augment=False,
    max_det=300,
    project="runs/native",
    name="native_predict_recheck",
    save_txt=True,
    save_conf=True,
    stream=True,
)


# ----------------------------------------------------------------------
# Convert native model.predict() Results -> canonical PredictionResult
# ----------------------------------------------------------------------

images = []
boxes = []
scores = []
labels = []

for native_result in native_results:
    image_height, image_width = native_result.orig_shape

    image_boxes = []
    image_scores = []
    image_labels = []

    if native_result.boxes is not None and len(native_result.boxes) > 0:
        xyxy = native_result.boxes.xyxy.detach().cpu().numpy()
        confs = native_result.boxes.conf.detach().cpu().numpy()
        classes = native_result.boxes.cls.detach().cpu().numpy()

        for box, score, class_id in zip(
            xyxy,
            confs,
            classes,
        ):
            x1, y1, x2, y2 = box

            image_boxes.append(
                [
                    float(x1 / image_width),
                    float(y1 / image_height),
                    float(x2 / image_width),
                    float(y2 / image_height),
                ]
            )
            image_scores.append(float(score))
            image_labels.append(int(class_id))

    images.append(
        str(Path(native_result.path).expanduser().resolve())
    )
    boxes.append(image_boxes)
    scores.append(image_scores)
    labels.append(image_labels)


prediction_result = PredictionResult(
    images=images,
    boxes=boxes,
    scores=scores,
    labels=labels,
    metadata={
        "source": "direct_ultralytics_model_predict",
        "weight": WEIGHT,
        "imgsz": 640,
        "batch": 32,
        "device": "0",
        "conf": 0.001,
        "iou": 0.7,
        "agnostic_nms": False,
        "augment": False,
        "max_det": 300,
    },
)

PREDICTION_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

prediction_result.save_npz(
    PREDICTION_OUTPUT
)

print()
print("=" * 80)
print("DIRECT ULTRALYTICS PREDICTION COMPLETE")
print("=" * 80)
print(f"Images:       {len(prediction_result.images)}")
print(f"Detections:   {prediction_result.num_detections}")
print(f"Saved cache:  {PREDICTION_OUTPUT}")


# ----------------------------------------------------------------------
# Load dataset metadata and ground truth
# ----------------------------------------------------------------------

with DATA_YAML.open("r", encoding="utf-8") as file:
    dataset = yaml.safe_load(file)

ground_truths = CustomValidator.load_ground_truths(
    prediction_result.images,
    labels_dir=LABELS_DIR,
    allow_missing_files=False,
)

gt_instance_count = sum(
    len(record["cls"])
    for record in ground_truths.values()
)


# ----------------------------------------------------------------------
# Custom validation
# ----------------------------------------------------------------------

VALIDATION_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

validator = CustomValidator(
    names=dataset["names"],
    save_dir=VALIDATION_OUTPUT_DIR,
    plots=True,
    min_conf=0.001,
    max_det=300,
    confusion_conf=0.25,
    confusion_iou=0.45,
    device="cpu",
)

validation_result = validator.evaluate(
    prediction_result,
    ground_truths,
    ground_truth_box_format="xywhn",
    ground_truth_label_offset=0,
)

overall = dict(validation_result.overall)

print()
print("=" * 80)
print("CUSTOM VALIDATION OF DIRECT model.predict() OUTPUT")
print("=" * 80)
print(f"GT instances:            {gt_instance_count}")

for key in (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "fitness",
):
    if key in overall:
        print(
            f"{key:24s}: "
            f"{float(overall[key]):.6f}"
        )

f1_peak = extract_smoothed_f1_peak(
    validator
)

if f1_peak is not None:
    peak_f1, peak_conf = f1_peak

    print(
        "F1 curve peak:           "
        f"{peak_f1:.6f} "
        f"at confidence {peak_conf:.6f}"
    )

print(
    f"Validation outputs:       "
    f"{VALIDATION_OUTPUT_DIR}"
)
print("=" * 80)