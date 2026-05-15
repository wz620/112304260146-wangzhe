import os
import pandas as pd
from ultralytics import YOLO

def get_image_id(filename):
    return filename

def predict_and_submit():
    model_path = 'runs/detect/runs/traffic_signs_v2/train/weights/best.pt'
    if not os.path.exists(model_path):
        model_path = 'runs/detect/runs/traffic_signs_v2/train/weights/last.pt'
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    test_images_dir = '第4次实验数据及提交格式/test/images'
    image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png'))])
    
    print(f"Found {len(image_files)} test images")
    
    results_list = []
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(test_images_dir, img_file)
        
        results = model.predict(img_path, conf=0.1, iou=0.5, verbose=False, device='0')
        
        image_id = get_image_id(img_file)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                x_center, y_center, width, height = box.xywhn[0].tolist()
                
                results_list.append({
                    'image_id': image_id,
                    'class_id': cls,
                    'x_center': round(x_center, 6),
                    'y_center': round(y_center, 6),
                    'width': round(width, 6),
                    'height': round(height, 6),
                    'confidence': round(conf, 6)
                })
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images...")
    
    df = pd.DataFrame(results_list)
    df = df.sort_values(by=['image_id', 'confidence'], ascending=[True, False])
    
    output_path = 'submission.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Total predictions: {len(df)}")

if __name__ == '__main__':
    predict_and_submit()
