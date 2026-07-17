import argparse
import os
import queue
import subprocess
import time
from pathlib import Path
from typing import List, Dict

OPTION_A_FULL_FINE_TUNE = {
    "option": "A_full_orddc",
    "data": DATA,
    "weights": ORDDC_WEIGHTS,
    "project": PROJECT,
    "epochs": 100,
    "batch": 32,
    "imgsz": 640,
    "lr0": 0.001,
    "lrf": 0.01,
    "optimizer": "SGD",
    "freeze": "0",
    "tag": "lr1e-3",
}

def build_command(job: Dict, train_script: str, device: str) -> List[str]:
    cmd = [
        "python", train_script,
        "--option", job["option"],
        "--data", job["data"],
        "--weights", job["weights"],
        "--project", job["project"],
        "--epochs", str(job.get("epochs", 100)),
        "--imgsz", str(job.get("imgsz", 640)),
        "--batch", str(job.get("batch", 32)),
        "--lr0", str(job.get("lr0", 0.001)),
        "--lrf", str(job.get("lrf", 0.01)),
        "--optimizer", job.get("optimizer", "SGD"),
        "--patience", str(job.get("patience", 25)),
        "--save-period", str(job.get("save_period", 25)),
        "--seed", str(job.get("seed", 42)),
        # "--deterministic" if job.get("deterministic", False) else "",
        # "--amp" if job.get("amp", False) else "",
        # "--exist-ok" if job.get("exist_ok", False) else "",
        "--hsv-h", str(job.get("hsv_h", 0.015)),
        "--hsv-s", str(job.get("hsv_s", 0.7)),
        "--hsv-v", str(job.get("hsv_v", 0.4)),
        "--degrees", str(job.get("degrees", 60)),
        "--translate", str(job.get("translate", 0.5)),
        "--scale", str(job.get("scale", 0.5)),
        "--shear", str(job.get("shear", 10.0)),
        "--perspective", str(job.get("perspective", 0.0005)),
        "--fliplr", str(job.get("fliplr", 0.5)),
        "--flipud", str(job.get("flipud", 0.0)),
        "--device", device,
        "--freeze", job.get("freeze", "0")
    ]
    
    if job.get("amp", False):
        cmd.append("--amp")
    if job.get("deterministic", False):
        cmd.append("--deterministic")
    if job.get("exist_ok", False):
        cmd.append("--exist-ok")
        
    if job.get("tag", ""):
        cmd.extend(["--tag", job["tag"]])
    
    if job.get("two_stage", False):
        cmd.append("--two-stage")
        cmd.append("--stage1-epochs")
        cmd.append(str(job.get("stage1_epochs", 30)))
        cmd.append("--stage2-epochs")
        cmd.append(str(job.get("stage2_epochs", 70)))
        cmd.append("--stage1-freeze")
        cmd.append(job.get("stage1_freeze", "head_only"))
        cmd.append("--stage2-freeze")
        cmd.append(job.get("stage2_freeze", "0"))
        if job.get("stage1_lr0") is not None:
            cmd.append("--stage1-lr0")
            cmd.append(str(job["stage1_lr0"]))
        if job.get("stage2_lr0") is not None:
            cmd.append("--stage2-lr0")
            cmd.append(str(job["stage2_lr0"]))
        
    return cmd

def launch_jobs(job: Dict, train_script: str, device: str):
    cmd = build_command(job, train_script, device)
    print(f"Launching job with command: {' '.join(cmd)}")
    subprocess.run(cmd)