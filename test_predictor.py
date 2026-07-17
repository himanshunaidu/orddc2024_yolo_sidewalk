
import yaml
from orddc2024.predictors.ultralytics_predictor import UltralyticsPredictor
# from orddc2024.predictors.yolov5_predictor import Yolov5Predictor
from orddc2024.predictors.ensemble_predictor import EnsemblePredictor

predictor_registry = {
    "ultralytics": UltralyticsPredictor,
    # "yolov5": Yolov5Predictor,
}
YAML_FILE = "./models/model_ph1.yaml"
IMAGE_PATH = "./data/RDD2022/China_Drone/train/images"
def load_yaml_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
CONFIG = load_yaml_config(YAML_FILE)
ultra_models_params = CONFIG['models'].get('yolov8', [])
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
predictor.load(models_params=None, images_path=IMAGE_PATH)
boxes, scores, labels = predictor.predict(batch_size=32)