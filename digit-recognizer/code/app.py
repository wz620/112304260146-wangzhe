import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
import numpy as np


class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


model_path = "model.pth"

model = DigitCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()


def preprocess_image(img: Image.Image) -> torch.Tensor:
    if img.mode != 'L':
        img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).reshape(1, 1, 28, 28)
    return img_tensor


def predict(img: Image.Image) -> dict:
    if img is None:
        return {"error": "Please upload an image"}
    
    img_tensor = preprocess_image(img)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        confidence = probs[0, pred].item()
    
    top3_probs, top3_indices = torch.topk(probs, 3, dim=1)
    top3_probs = top3_probs[0].tolist()
    top3_indices = top3_indices[0].tolist()
    
    result = {
        "Prediction": str(pred),
        "Confidence": f"{confidence * 100:.2f}%"
    }
    
    for i, (idx, prob) in enumerate(zip(top3_indices, top3_probs)):
        result[f"Top-{i+1}"] = f"{idx} ({prob*100:.1f}%)"
    
    return result


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Handwritten Digit Image"),
    outputs=gr.Label(label="Prediction Result"),
    title="Handwritten Digit Recognition",
    description="Upload an image of a handwritten digit (0-9), and the CNN model will predict the digit.",
    examples=None,
)

demo.launch()
