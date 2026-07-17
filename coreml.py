"""
This script converts a YOLOv5 model to CoreML format.
Currently, it uses the old TorchScript method for conversion.
"""
import argparse
import os.path as osp
import sys
import yaml

import numpy as np
import torch
import torch.nn as nn
import torchvision
import json
import cv2
from PIL import Image

from ultralytics import YOLO
import coremltools as ct

def load_yaml_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

class WrappedYoloV8:
    def __init__(self, models_params):
        self.framework = "yolov8"
        self.models_params = models_params
        weight = models_params['weight']
        self.model = YOLO(weight)
        # self.model.eval()
    
    def export(self):
        self.model.export(format="coreml", imgsz=self.models_params.get('imgsz', 640), nms=True)
    
if __name__ == "__main__":
    # yaml_file = "./model_ph2.yaml"
    yaml_file = "./train_scripts/model_lab_train.yaml"
    
    config = load_yaml_config(yaml_file)
    ultra_models_params = config['models'].get('yolov8', [])
    
    model = WrappedYoloV8(ultra_models_params[0])
    model.export()