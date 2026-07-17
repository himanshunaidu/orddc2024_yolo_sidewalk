# Train Commands

## Train Yolov8x on ORDDC Dataset

(Note: Even for training, we can use the fine-tuning script, we will use the coco-pretrained weights for training from scratch.)

python yolov8_finetune.py \
  --data /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/RDD2022/global_train.yaml \
  --weights /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/models_ph2/yolov8n_coco.pt \
  --project /rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/orddc \
  --epochs 1000 \
  --imgsz 640 \
  --batch 128 \
  --lr0 0.004 \
  --lrf 0.01 \
  --patience 100 \
  --optimizer SGD \
  --freeze 0 \
  --option A_orddc \
  --tag "lr4e-3" \
  --device 0