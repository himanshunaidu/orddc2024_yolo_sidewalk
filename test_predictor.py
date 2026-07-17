
import yaml
from pathlib import Path
from orddc2024.predictors.ultralytics_predictor import UltralyticsPredictor
# from orddc2024.predictors.yolov5_predictor import Yolov5Predictor
from orddc2024.predictors.ensemble_predictor import EnsemblePredictor

from evaluation.validator import CustomValidator

predictor_registry = {
    "ultralytics": UltralyticsPredictor,
    # "yolov5": Yolov5Predictor,
}
MODEL_YAML_FILE = "./models/model_ph1.yaml"
DATA_YAML_FILE = "./data/sample/train.yaml"
# IMAGE_PATH = "./data/RDD2022/China_Drone/train/images"
# IMAGE_PATH = "./data/sample/train/images"
def load_yaml_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
model_config = load_yaml_config(MODEL_YAML_FILE)
data_config = load_yaml_config(DATA_YAML_FILE)

ultra_models_params = model_config['models'].get('yolov8', [])
for index, model_param in enumerate(ultra_models_params):
    model_param['framework'] = 'ultralytics'
# yolov5_weights_params = CONFIG['models'].get('yolov5', [])
models_params = ultra_models_params  # + yolov5_weights_params
predictor = EnsemblePredictor(
    framework="yolo_ensemble",
    models_params=models_params,
    predictor_registry=predictor_registry,
    fusion_method="wbf",
    iou_thr=0.55,
    skip_box_thr=0.001,
    conf_type="avg",
    max_workers=2,
)

data_file_path = data_config['path']
val_file_path = data_config['val']
val_file = Path(data_file_path) / val_file_path

class_names = data_config['names']

predictor.load(models_params=None, images_path=val_file)
boxes, scores, labels = predictor.predict(batch_size=32)

images = predictor.images
ground_truths = CustomValidator.load_ground_truths(images=images, labels_dir=None, allow_missing_files=False)
# print(ground_truths)

predictions = CustomValidator.predictions_from_lists(
    images=images,
    boxes=boxes,
    scores=scores,
    labels=labels,
)
# print(predictions)

validator = CustomValidator(
    names=class_names,
    save_dir="eval_runs/test_predictor/",
    device="cuda"
)
result = validator.evaluate(
    images=images,
    ground_truths=ground_truths,
    predictions=predictions,
    ground_truth_box_format="xywhn",
    prediction_box_format="xyxyn",

    # Your current UltralyticsPredictor returns cls + 1.
    prediction_label_offset=0,
)
print(result.overall)
print(result.confusion_matrix)