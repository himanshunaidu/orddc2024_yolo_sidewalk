from __future__ import annotations

import argparse
import json
from pathlib import Path

# Imported only inside the dedicated YOLOv8 subprocess/environment.
from ultralytics import YOLO


def normalize_box(box, img_width, img_height):
    x1, y1, x2, y2 = box
    return [
        x1 / img_width,
        y1 / img_height,
        x2 / img_width,
        y2 / img_height,
    ]


def predict_one_model(model_param, images):
    weight = model_param["weight"]
    conf = float(model_param.get("conf", 0.001))
    iou = float(model_param.get("iou", 0.7))
    imgsz = int(model_param.get("img_size", model_param.get("imgsz", 640)))
    augment = bool(model_param.get("augment", False))
    agnostic_nms = bool(model_param.get("agnostic_nms", False))

    # Keep original ORDDC convention by default (classes 1..N).
    # Set label_offset=0 in the model config to use zero-based classes.
    label_offset = int(model_param.get("label_offset", 1))

    predict_kwargs = {
        "source": images,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "augment": augment,
        "agnostic_nms": agnostic_nms,
        "verbose": False,
        "stream": False,
    }

    for key in ("device", "max_det", "half", "batch"):
        if key in model_param and model_param[key] is not None:
            predict_kwargs[key] = model_param[key]

    print("-" * 72)
    print(f"Loading YOLOv8 model: {weight}")
    model = YOLO(weight)
    print(f"Predicting {len(images)} images")

    results = model.predict(**predict_kwargs)

    if len(results) != len(images):
        raise RuntimeError(
            f"Expected {len(images)} results but received {len(results)}."
        )

    model_boxes = []
    model_scores = []
    model_labels = []

    for result in results:
        image_height, image_width = result.orig_shape

        boxes = []
        scores = []
        labels = []

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            confs = result.boxes.conf.detach().cpu().tolist()
            classes = result.boxes.cls.detach().cpu().tolist()

            for box, score, class_id in zip(xyxy, confs, classes):
                boxes.append(normalize_box(box, image_width, image_height))
                scores.append(float(score))
                labels.append(int(class_id) + label_offset)

        model_boxes.append(boxes)
        model_scores.append(scores)
        model_labels.append(labels)

    return model_boxes, model_scores, model_labels


def run(request):
    images = request["images"]
    models_params = request["models"]

    boxes_list = []
    scores_list = []
    labels_list = []

    # Sequential on one GPU; one subprocess can serve any number of YOLOv8 models.
    for index, model_param in enumerate(models_params, start=1):
        print(
            f"YOLOv8 model {index}/{len(models_params)}: "
            f"{model_param['weight']}"
        )
        boxes, scores, labels = predict_one_model(model_param, images)
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    return {
        "images": images,
        "boxes": boxes_list,
        "scores": scores_list,
        "labels": labels_list,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    request_path = Path(args.request)
    output_path = Path(args.output)

    request = json.loads(request_path.read_text(encoding="utf-8"))
    result = run(request)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result), encoding="utf-8")

    print(f"YOLOv8 predictions written to: {output_path}")


if __name__ == "__main__":
    main()