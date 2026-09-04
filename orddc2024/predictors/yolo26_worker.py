# YOLO26 worker script for the ORDDC2024 project. Performs inference using the YOLOv8 model.
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO
from tqdm import tqdm


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
    
    batch_size = int(model_param.get("batch", 1))

    predict_kwargs = {
        # "source": images,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "augment": augment,
        "agnostic_nms": agnostic_nms,
        "verbose": False,
        "stream": True,
    }

    for key in ("device", "max_det", "half", "batch"):
        if key in model_param and model_param[key] is not None:
            predict_kwargs[key] = model_param[key]

    print(f"Loading YOLOv8 model: {weight}")
    model = YOLO(weight)
    # results = model.predict(**predict_kwargs)

    # if len(results) != len(images):
    #     raise RuntimeError(
    #         f"Expected {len(images)} results but received {len(results)}."
    #     )

    model_boxes = []
    model_scores = []
    model_labels = []
    
    total_results = 0
    for start_idx in tqdm(range(0, len(images), batch_size), desc="Processing batches"):
        batch_images = images[start_idx:start_idx + batch_size]
        results = model.predict(source=batch_images, **predict_kwargs)

        batch_result_count = 0
        
        for result in results:
            image_height, image_width = result.orig_shape
            boxes, scores, labels = [], [], []

            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.detach().cpu().tolist()
                confs = result.boxes.conf.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()

                for box, score, class_id in zip(xyxy, confs, classes):
                    boxes.append(
                        normalize_box(box, image_width, image_height)
                    )
                    scores.append(float(score))
                    # Canonical PredictionResult contract: zero-based labels.
                    labels.append(int(class_id))

            model_boxes.append(boxes)
            model_scores.append(scores)
            model_labels.append(labels)
            total_results += 1
            batch_result_count += 1
        # if batch_result_count != batch_size:
        #     raise RuntimeError(
        #         f"Expected batch size of {batch_size} but received {batch_result_count}."
        #     )
    if total_results != len(images):
        raise RuntimeError(
            f"Expected {len(images)} results but received {total_results}."
        )
    
    return model_boxes, model_scores, model_labels


def run(request):
    images = request["images"]

    boxes_list = []
    scores_list = []
    labels_list = []

    for model_param in request["models"]:
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

    request = json.loads(
        Path(args.request).read_text(encoding="utf-8")
    )
    result = run(request)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()