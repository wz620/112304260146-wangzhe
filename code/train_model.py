from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m.pt')
    
    results = model.train(
        data='code/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        device='0',
        project='runs/traffic_signs_v2',
        name='train',
        exist_ok=True,
        patience=15,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        workers=4,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
    )
    
    print("Training completed!")
    print(f"Results saved to: runs/traffic_signs_v2/train/")
