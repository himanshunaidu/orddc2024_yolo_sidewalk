from __future__ import annotations

import argparse
import json
from pathlib import Path

# IMPORTANT:
# This import happens only inside the dedicated YOLO26 subprocess.
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

    # Optional settings are forwarded only when explicitly configured.
    for key in ("device", "max_det", "half", "batch"):
        if key in model_param and model_param[key] is not None:
            predict_kwargs[key] = model_param[key]

    print("-" * 72)
    print(f"Loading YOLO26 model: {weight}")
    model = YOLO(weight)

    print(
        f"Predicting {len(images)} images "
        f"(imgsz={imgsz}, conf={conf}, iou={iou})"
    )

    results = model.predict(**predict_kwargs)

    model_boxes = []
    model_scores = []
    model_labels = []

    if len(results) != len(images):
        raise RuntimeError(
            f"Expected {len(images)} prediction results but received {len(results)}."
        )

    for image_path, result in zip(images, results):
        image_height, image_width = result.orig_shape

        boxes = []
        scores = []
        labels = []

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            confs = result.boxes.conf.detach().cpu().tolist()
            classes = result.boxes.cls.detach().cpu().tolist()

            for box, score, class_id in zip(xyxy, confs, classes):
                boxes.append(
                    normalize_box(
                        box,
                        img_width=image_width,
                        img_height=image_height,
                    )
                )
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

    # Deliberately sequential inside one backend process. This avoids having
    # several models contend for the same GPU while still amortizing process
    # startup and Ultralytics import overhead across all YOLO26 models.
    for model_index, model_param in enumerate(models_params, start=1):
        print(
            f"YOLO26 model {model_index}/{len(models_params)}: "
            f"{model_param['weight']}"
        )

        boxes, scores, labels = predict_one_model(
            model_param=model_param,
            images=images,
        )

        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    return {
        "images": images,
        "boxes": boxes_list,
        "scores": scores_list,
        "labels": labels_list,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    request_path = Path(args.request)
    output_path = Path(args.output)

    request = json.loads(request_path.read_text(encoding="utf-8"))
    result = run(request)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result),
        encoding="utf-8",
    )

    print("=" * 72)
    print(f"YOLO26 predictions written to: {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()