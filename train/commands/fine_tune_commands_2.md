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

