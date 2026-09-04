# Fine-tuning Commands 2

## 1 Epoch training to get quick validation results for existing models

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/RDD2022/global_train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/models_ph2/v8n_175_16_960.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/orddc \
  --epochs 1 \
  --imgsz 960 \
  --batch 16 \
  --lr0 0.001 \
  --lrf 0.01 \
  --patience 100 \
  --optimizer SGD \
  --freeze 0 \
  --option orddc_175_16_960 \
  --tag "lr1e-3" \
  --device 0

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/RDD2022/global_train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/models_ph2/v5s_best_640_025.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/orddc \
  --epochs 1 \
  --imgsz 960 \
  --batch 16 \
  --lr0 0.001 \
  --lrf 0.01 \
  --patience 100 \
  --optimizer SGD \
  --freeze 0 \
  --option orddc_175_16_960 \
  --tag "lr1e-3" \
  --device 0

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/RDD2022/global_train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/models/models_ph1/v8x_16_100.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/orddc \
  --epochs 0 \
  --imgsz 960 \
  --batch 16 \
  --lr0 0.001 \
  --lrf 0.01 \
  --patience 100 \
  --optimizer SGD \
  --freeze 0 \
  --option orddc_175_16_960 \
  --tag "lr1e-3" \
  --device 0

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/models_ph2/v8n_175_16_960.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk \
  --epochs 1000 \
  --imgsz 960 \
  --batch 128 \
  --lr0 0.004 \
  --lrf 0.01 \
  --patience 100 \
  --optimizer SGD \
  --freeze 0 \
  --option A_full_soft_5 \
  --tag "lr4e-3" \
  --device 2

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/models_ph2/v8n_175_16_960.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk \
  --epochs 1 \
  --imgsz 960 \
  --batch 1 \
  --lr0 0.004 \
  --lrf 0.01 \
  --patience 100 \
  --optimizer SGD \
  --freeze 0 \
  --option A_full_soft_5_2 \
  --tag "lr4e-3" \
  --device 2

# Fine-tuning Commands 3

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk/A_full_orddc_weights_v8n_175_16_960_batch_128_lr0_0.004_lrf_0.01_imgsz_960_opt_SGD_lr4e-33/weights/v8n_175_16_960.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk \
  --epochs 200 \
  --imgsz 960 \
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
  --epochs 200 \
  --imgsz 640 \
  --batch 16 \
  --lr0 0.001 \
  --lrf 0.01 \
  --patience 100 \
  --optimizer SGD \
  --freeze 0 \
  --option A_masked_2_full_orddc \
  --tag "lr1e-3" \
  --device 1