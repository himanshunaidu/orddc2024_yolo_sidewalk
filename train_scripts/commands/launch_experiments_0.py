import argparse
import os
import queue
import subprocess
import time
from pathlib import Path
from typing import List, Dict


def build_command(job: Dict, train_script: str) -> List[str]:
    cmd = [
        "python", train_script,
        "--data", job["data"],
        "--weights", job["weights"],
        "--project", job.get("project", "runs/detect_transfer"),
        "--option", job["option"],
        "--epochs", str(job.get("epochs", 100)),
        "--imgsz", str(job.get("imgsz", 640)),
        "--batch", str(job.get("batch", 32)),
        "--lr0", str(job.get("lr0", 0.001)),
        "--lrf", str(job.get("lrf", 0.01)),
        "--optimizer", job.get("optimizer", "SGD"),
        "--freeze", str(job.get("freeze", "0")),
        "--device", "0",
        "--workers", str(job.get("workers", 8)),
        "--patience", str(job.get("patience", 25)),
        "--save-period", str(job.get("save_period", 25)),
        "--seed", str(job.get("seed", 42)),
        "--close-mosaic", str(job.get("close_mosaic", 10)),
        "--mosaic", str(job.get("mosaic", 1.0)),
        "--mixup", str(job.get("mixup", 0.0)),
        "--degrees", str(job.get("degrees", 0.0)),
        "--translate", str(job.get("translate", 0.1)),
        "--scale", str(job.get("scale", 0.5)),
        "--fliplr", str(job.get("fliplr", 0.5)),
        "--flipud", str(job.get("flipud", 0.0)),
    ]

    if job.get("amp", True):
        cmd.append("--amp")

    if job.get("deterministic", False):
        cmd.append("--deterministic")

    if job.get("exist_ok", False):
        cmd.append("--exist-ok")

    if job.get("tag", ""):
        cmd.extend(["--tag", job["tag"]])

    if job.get("two_stage", False):
        cmd.append("--two-stage")
        cmd.extend(["--stage1-epochs", str(job.get("stage1_epochs", 30))])
        cmd.extend(["--stage2-epochs", str(job.get("stage2_epochs", 70))])
        cmd.extend(["--stage1-freeze", str(job.get("stage1_freeze", "head_only"))])
        cmd.extend(["--stage2-freeze", str(job.get("stage2_freeze", "0"))])

        if "stage1_lr0" in job:
            cmd.extend(["--stage1-lr0", str(job["stage1_lr0"])])

        if "stage2_lr0" in job:
            cmd.extend(["--stage2-lr0", str(job["stage2_lr0"])])

    return cmd


def launch_jobs(jobs: List[Dict], gpus: List[int], train_script: str, log_dir: str):
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    job_queue = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    running = {}

    while not job_queue.empty() or running:
        # Fill available GPUs
        for gpu in gpus:
            if gpu in running:
                continue

            if job_queue.empty():
                continue

            job = job_queue.get()
            option = job["option"]
            tag = job.get("tag", "")
            log_file = log_path / f"gpu{gpu}_{option}_{tag}_{int(time.time())}.log"

            cmd = build_command(job, train_script=train_script)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

            print("=" * 88)
            print(f"Launching job on physical GPU {gpu}")
            print(f"Option: {option}")
            print(f"Tag: {tag}")
            print(f"Log: {log_file}")
            print("Command:")
            print(" ".join(cmd))
            print("=" * 88)

            f = open(log_file, "w")

            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )

            running[gpu] = {
                "process": process,
                "log_file_handle": f,
                "job": job,
                "log_file": log_file,
            }

        # Poll running jobs
        finished_gpus = []

        for gpu, record in running.items():
            process = record["process"]
            return_code = process.poll()

            if return_code is not None:
                record["log_file_handle"].close()

                job = record["job"]
                print("-" * 88)
                print(f"Finished job on GPU {gpu}")
                print(f"Option: {job['option']}")
                print(f"Tag: {job.get('tag', '')}")
                print(f"Return code: {return_code}")
                print(f"Log: {record['log_file']}")
                print("-" * 88)

                finished_gpus.append(gpu)

        for gpu in finished_gpus:
            del running[gpu]

        time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-script", type=str, default="train_yolov8_transfer.py")
    parser.add_argument("--gpus", type=str, required=True, help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--log-dir", type=str, default="experiment_logs")
    args = parser.parse_args()

    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]

    DATA = "/path/to/your/data.yaml"
    ORDDC_WEIGHTS = "/path/to/orddc_yolov8n_best.pt"
    COCO_WEIGHTS = "yolov8n.pt"
    PROJECT = "runs/sidewalk_ground_disruption_yolov8"

    jobs = [
        # Option A: ORDDC -> full fine-tune
        {
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
        },
        {
            "option": "A_full_orddc",
            "data": DATA,
            "weights": ORDDC_WEIGHTS,
            "project": PROJECT,
            "epochs": 100,
            "batch": 32,
            "imgsz": 640,
            "lr0": 0.0005,
            "lrf": 0.01,
            "optimizer": "SGD",
            "freeze": "0",
            "tag": "lr5e-4",
        },

        # Option B1: head-only
        {
            "option": "B_head_only_orddc",
            "data": DATA,
            "weights": ORDDC_WEIGHTS,
            "project": PROJECT,
            "epochs": 100,
            "batch": 32,
            "imgsz": 640,
            "lr0": 0.001,
            "lrf": 0.01,
            "optimizer": "SGD",
            "freeze": "head_only",
            "tag": "head_only",
        },

        # Option B2: freeze first N layers. You should tune this after printing model layers.
        {
            "option": "B_partial_freeze_orddc",
            "data": DATA,
            "weights": ORDDC_WEIGHTS,
            "project": PROJECT,
            "epochs": 100,
            "batch": 32,
            "imgsz": 640,
            "lr0": 0.001,
            "lrf": 0.01,
            "optimizer": "SGD",
            "freeze": "10",
            "tag": "freeze10",
        },

        # Option C1: two-stage head-only warmup -> full fine-tune
        {
            "option": "C_two_stage_orddc",
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
            "two_stage": True,
            "stage1_epochs": 30,
            "stage2_epochs": 70,
            "stage1_freeze": "head_only",
            "stage2_freeze": "0",
            "stage1_lr0": 0.001,
            "stage2_lr0": 0.0005,
            "tag": "head_to_full",
        },

        # Option C2: freeze most layers first -> partial/full unfreeze
        {
            "option": "C_two_stage_orddc",
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
            "two_stage": True,
            "stage1_epochs": 30,
            "stage2_epochs": 70,
            "stage1_freeze": "10",
            "stage2_freeze": "0",
            "stage1_lr0": 0.001,
            "stage2_lr0": 0.0005,
            "tag": "freeze10_to_full",
        },

        # Baseline: COCO -> full fine-tune
        {
            "option": "D_full_coco",
            "data": DATA,
            "weights": COCO_WEIGHTS,
            "project": PROJECT,
            "epochs": 100,
            "batch": 32,
            "imgsz": 640,
            "lr0": 0.001,
            "lrf": 0.01,
            "optimizer": "SGD",
            "freeze": "0",
            "tag": "coco_baseline",
        },
    ]

    launch_jobs(
        jobs=jobs,
        gpus=gpus,
        train_script=args.train_script,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()