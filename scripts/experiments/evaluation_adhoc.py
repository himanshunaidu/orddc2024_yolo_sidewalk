from ultralytics import YOLO

model = YOLO("/rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/runs/iospm_surface_integrity_orddc_sidewalk/A_full_orddc_weights_v8n_175_16_960_batch_128_lr0_0.004_lrf_0.01_imgsz_960_opt_SGD_lr4e-32/weights/best.pt")

model.val(
    data="/rightofwai/rightofwai/homes/hnaidu36/iospointmapper/SurfaceIntegrity/orddc2024/data/iOSPointMapper_Surface_Integrity_ORDDC_Sidewalk/train.yaml",
    imgsz=960,
    batch=32,
    device="0",
    conf=0.001,
    iou=0.7,
    agnostic_nms=False,
    augment=False,
    max_det=300,
    plots=True,
    project="runs/native",
    name="native_val_recheck11",
    save_txt=True,
    save_conf=True,
    rect=False,
)