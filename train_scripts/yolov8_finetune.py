import os
import time
import torch
import json
from pathlib import Path
from ultralytics import YOLO
import argparse
from typing import Optional

# FINE_TUNE_OPTIONS = ["fine_tune_all", "freeze_backbone", "adapter"]

def parse_freeze_arg(value: str):
    """
    Accepts:
        "0"             -> 0
        "10"            -> 10
        "none"          -> None
        "head_only"     -> special mode handled later
        "0,1,2,3"       -> [0, 1, 2, 3]
    """
    value = str(value).strip().lower()

    if value in {"none", "null"}:
        return None

    if value == "head_only":
        return "head_only"

    if "," in value:
        return [int(x.strip()) for x in value.split(",") if x.strip()]

    return int(value)


def get_num_model_layers(model: YOLO) -> int:
    """
    Ultralytics YOLO models usually expose the layer list at model.model.model.
    This helper keeps the logic in one place.
    """
    return len(model.model.model)

def resolve_freeze_setting(model: YOLO, freeze_arg):
    """
    Converts special freeze modes into Ultralytics-compatible freeze values.
    """
    num_layers = get_num_model_layers(model)
    if freeze_arg == "head_only":
        # Usually the final layer is the Detect head.
        # Freezing all earlier layers leaves the detection head trainable.
        return list(range(num_layers - 1))

    return freeze_arg 

def build_run_name(args, stage_name: Optional[str] = None):
    weights_name = Path(args.weights).stem

    parts = [
        args.option,
        f"weights_{weights_name}",
        f"batch_{args.batch}",
        f"lr0_{args.lr0}",
        f"lrf_{args.lrf}",
        f"imgsz_{args.imgsz}",
        # f"freeze_{str(args.freeze).replace(',', '-')}",
        f"opt_{args.optimizer}",
    ]

    if stage_name is not None:
        parts.insert(1, stage_name)

    if args.tag:
        parts.append(args.tag)

    return "_".join(parts)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--option",
        type=str,
        # choices=FINE_TUNE_OPTIONS,
        default="A_full_orddc",
        help="Fine-tuning option to use."
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML."
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Initial weights, e.g. yolov8n.pt or path/to/best.pt."
    )

    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect_train",
        help="Output project directory."
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="SGD")
    
    parser.add_argument("--patience", type=int, default=25, help="Patience for early stopping.")
    parser.add_argument("--save-period", type=int, default=25, help="Period (in epochs) to save model weights.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training for reproducibility.")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (AMP) for faster training and lower memory usage.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow existing project directory to be overwritten.")
    
    # Augmentation
    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV-Hue augmentation factor.")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV-Saturation augmentation factor.")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV-Value augmentation factor.")
    # parser.add_argument("--close-mosaic", type=int, default=10)
    # parser.add_argument("--mosaic", type=float, default=1.0)
    # parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--degrees", type=float, default=60, help="Degrees for rotation augmentation.")
    parser.add_argument("--translate", type=float, default=0.5, help="Translation factor for augmentation.")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale factor for augmentation.")
    parser.add_argument("--shear", type=float, default=10.0, help="Shear factor for augmentation.")
    parser.add_argument("--perspective", type=float, default=0.0005, help="Perspective factor for augmentation.")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability for augmentation.")
    parser.add_argument("--flipud", type=float, default=0.0, help="Vertical flip probability for augmentation.")
    
    # Two stage mode (useful for adapters or freezing backbone)
    parser.add_argument("--two-stage", action="store_true", help="Enable two-stage training (stage 1 and stage 2).")
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage2-epochs", type=int, default=70)
    parser.add_argument("--stage1-freeze", type=str, default="head_only")
    parser.add_argument("--stage2-freeze", type=str, default="0")
    parser.add_argument("--stage1-lr0", type=float, default=None)
    parser.add_argument("--stage2-lr0", type=float, default=None)
    
    parser.add_argument(
        "--freeze",
        type=str,
        default="0",
        help='0 for full fine-tune, "head_only", int N, or comma-separated layer indices.'
    )
    parser.add_argument("--device", type=str, default="0", help="Device to use for training (e.g., '0' for GPU 0, 'cpu' for CPU).")
    parser.add_argument("--tag", type=str, default="", help="Optional tag to append to the run name for easier identification.")

    return parser.parse_args()

def train_model_one_stage(
    *,
    args,
    weights: str,
    run_name: str,
    freeze,
    lr0: float,
    epochs: int,
    project: str,
):
    print(f"Starting training stage with run name: {run_name} {weights}")
    model = YOLO(weights)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    freeze_resolved = resolve_freeze_setting(model, freeze)
    
    # if args.print_layers:
    #     print_model_layers(model)
    
    print("=" * 72)
    print(f"Training with data={args.data}")
    print(f"Weights={args.weights}")
    print(f"Project={args.project}")
    print(f"Name={run_name}")
    print("=" * 72)
    
    start_time = time.time()
    
    results = model.train(
        data=args.data,
        task="detect",
        epochs=epochs,
        imgsz=args.imgsz,
        device=args.device if torch.cuda.is_available() else "cpu",
        batch=args.batch,
        project=project,
        name=run_name,
        optimizer=args.optimizer,
        lr0=lr0,
        lrf=args.lrf,
        cos_lr=True,
        freeze=freeze_resolved,
        patience=args.patience,
        save=True,
        save_period=args.save_period,
        exist_ok=args.exist_ok,
        seed=args.seed,
        deterministic=args.deterministic,
        amp=args.amp,
        # Augmentation parameters
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        # close_mosaic=args.close_mosaic,
        # mosaic=args.mosaic,
        # mixup=args.mixup,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        fliplr=args.fliplr,
        flipud=args.flipud,
    )
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    expected_dir = Path(project) / run_name
    weights_dir = expected_dir / "weights"
    best_weights_path = weights_dir / "best.pt"
    last_weights_path = weights_dir / "last.pt"
    
    metadata = {
        "run_name": run_name,
        "data": args.data,
        "weights": str(weights),
        "project": project,
        "epochs": epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "optimizer": args.optimizer,
        "lr0": lr0,
        "lrf": args.lrf,
        "freeze": str(freeze_resolved),
        "device": str(device),
        "elapsed_seconds": elapsed_time,
        "best_weights": str(best_weights_path),
        "last_weights": str(last_weights_path),
    }
    
    metadata_path = expected_dir / "training_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=4))
    
    print("#" * 72)
    print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Expected weights directory: {weights_dir}")
    print(f"Best weights path: {best_weights_path}")
    print(f"Last weights path: {last_weights_path}")
    print(f"Training metadata saved to: {metadata_path}")
    print("#" * 72)
    
    return str(best_weights_path), str(last_weights_path)

def main():
    args = parse_args()
    
    freeze = parse_freeze_arg(args.freeze)
    
    if not args.two_stage:
        run_name = build_run_name(args)
        train_model_one_stage(
            args=args,
            weights=args.weights,
            run_name=run_name,
            freeze=freeze,
            lr0=args.lr0,
            epochs=args.epochs,
            project=args.project,
        )
        return
    
    stage1_freeze = parse_freeze_arg(args.stage1_freeze)
    stage2_freeze = parse_freeze_arg(args.stage2_freeze)
    
    stage1_lr0 = args.stage1_lr0 if args.stage1_lr0 is not None else args.lr0
    stage2_lr0 = args.stage2_lr0 if args.stage2_lr0 is not None else args.lr0 * 0.5
    
    stage1_name = build_run_name(args, stage_name="stage1")
    best_stage1, last_stage1 = train_model_one_stage(
        args=args,
        weights=args.weights,
        run_name=stage1_name,
        freeze=stage1_freeze,
        lr0=stage1_lr0,
        epochs=args.stage1_epochs,
        project=args.project,
    )
    # Usually use best.pt from stage 1 as stage 2 initialization.
    stage2_name = build_run_name(args, stage_name="stage2")
    train_model_one_stage(
        args=args,
        weights=best_stage1,
        run_name=stage2_name,
        freeze=stage2_freeze,
        lr0=stage2_lr0,
        epochs=args.stage2_epochs,
        project=args.project,
    )
                    
if __name__ == "__main__":
    main()

# python yolov8_finetune.py \
#   --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
#   --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/models_ph2/v8n_175_16_960.pt \
#   --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk \
#   --epochs 100 \
#   --batch 32 \
#   --lr0 0.01