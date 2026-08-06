# Fine-tuning Commands

## Option A1 Full Fine-tune

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/models_ph2/v8n_175_16_960.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk \
  --epochs 1000 \
  --imgsz 640 \
  --batch 128 \
  --lr0 0.004 \
  --lrf 0.01 \
  --patience 100 \
  --optimizer SGD \
  --freeze 0 \
  --option A_full_orddc \
  --tag "lr4e-3" \
  --device 0

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/models_ph2/yolov8x.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk \
  --epochs 1000 \
  --imgsz 640 \
  --batch 16 \
  --lr0 0.001 \
  --lrf 0.01 \
  --patience 100 \
  --optimizer SGD \
  --freeze 0 \
  --option A_masked_2_full_orddc \
  --tag "lr1e-3" \
  --device 0

// ORDDC Training
iospointmapper/SurfaceIntegrity/orddc2024/data/RDD2022/train_Norway.yaml

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/models_ph2/v8n_175_16_960.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk \
  --epochs 1 \
  --imgsz 640 \
  --batch 32 \
  --lr0 0.001 \
  --lrf 0.01 \
  --patience 100 \
  --optimizer SGD \
  --freeze 0 \
  --option ORDDC_iOSPointMapper \
  --tag "lr1e-3" \
  --device 0

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/models_ph2/yolov8n_coco.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk \
  --epochs 1000 \
  --imgsz 640 \
  --batch 16 \
  --lr0 0.001 \
  --lrf 0.01 \
  --patience 100 \
  --optimizer SGD \
  --freeze 0 \
  --option A_masked_2_full \
  --tag "lr1e-3" \
  --device 0

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/models_ph2/v8n_175_16_960.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk \
  --epochs 1000 \
  --imgsz 640 \
  --batch 16 \
  --lr0 0.001 \
  --lrf 0.01 \
  --patience 100 \
  --optimizer SGD \
  --freeze 0 \
  --option A_masked_2_full \
  --tag "lr1e-3" \
  --device 0

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/models_ph2/yolov8x.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk \
  --epochs 1000 \
  --imgsz 640 \
  --batch 32 \
  --lr0 0.001 \
  --lrf 0.01 \
  --patience 100 \
  --optimizer SGD \
  --freeze 0 \
  --option A_full_soft_5 \
  --tag "lr1e-3" \
  --device 0

## Option A2 Full Fine-tune

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/models_ph2/v8n_175_16_960.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --lr0 0.0005 \
  --lrf 0.01 \
  --optimizer SGD \
  --freeze 0 \
  --option A_full_orddc \
  --tag "lr5e-4" \
  --device 1

## Option B1 Head Only

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/models_ph2/v8n_175_16_960.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --lr0 0.001 \
  --lrf 0.01 \
  --optimizer SGD \
  --freeze head_only \
  --option B_head_only_orddc \
  --tag "head_only" \
  --device 2

## Option B2 First N Layers

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/models_ph2/v8n_175_16_960.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --lr0 0.001 \
  --lrf 0.01 \
  --optimizer SGD \
  --freeze 10 \
  --option B_first_n_layers_orddc \
  --tag "first_10_layers" \
  --device 3

## Option C1 Two-stage Head-only warmup -> Full Fine-tune

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/models_ph2/v8n_175_16_960.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk \
  --epochs 1000 \
  --imgsz 640 \
  --batch 128 \
  --lr0 0.001 \
  --lrf 0.01 \
  --optimizer SGD \
  --freeze 0 \
  --two-stage \
  --stage1-freeze head_only \
  --stage1-lr0 0.005 \
  --stage1-epochs 300 \
  --stage2-freeze 0 \
  --stage2-lr0 0.001 \
  --stage2-epochs 700 \
  --option C_two_stage_orddc \
  --tag "head_to_full" \
  --device 3

## Option C2 Two-stage First N Layers warmup -> Full Fine-tune

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/models_ph2/v8n_175_16_960.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk \
  --epochs 1000 \
  --imgsz 640 \
  --batch 128 \
  --lr0 0.001 \
  --lrf 0.01 \
  --optimizer SGD \
  --freeze 0 \
  --two-stage \
  --stage1-freeze 10 \
  --stage1-lr0 0.005 \
  --stage1-epochs 300 \
  --stage2-freeze 0 \
  --stage2-lr0 0.001 \
  --stage2-epochs 700 \
  --option C_two_stage_orddc \
  --tag "freeze10_to_full" \
  --device 1
