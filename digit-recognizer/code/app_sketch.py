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


def predict_sketch(img: Image.Image) -> dict:
    if img is None:
        return {"error": "Please draw a digit"}
    
    img_tensor = preprocess_image(img)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        confidence = probs[0, pred].item()
    
    top3_probs, top3_indices = torch.topk(probs, 3, dim=1)
    top3_probs = top3_probs[0].tolist()
    top3_indices = top3_indices[0].tolist()
    
    labels = [str(idx) for idx in top3_indices]
    values = [prob * 100 for prob in top3_probs]
    
    return {
        str(pred): confidence,
    }, {labels[i]: values[i] for i in range(3)}


with gr.Blocks() as demo:
    gr.Markdown("# Handwritten Digit Recognition System")
    gr.Markdown("Draw a digit (0-9) on the canvas below, then click Submit to recognize.")
    
    with gr.Row():
        with gr.Column():
            sketch = gr.Sketchpad(
                label="Draw a digit",
                type="pil",
                canvas_size=(280, 280),
            )
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Submit")
        
        with gr.Column():
            result_label = gr.Label(label="Prediction Result")
            result_bar = gr.BarPlot(
                x=[0, 1, 2],
                y=[0, 0, 0],
                x_label="Digit",
                y_label="Probability (%)",
                label="Top-3 Probabilities",
                visible=False,
            )
    
    def clear_and_predict(img):
        if img is None:
            return None, {str(i): 0 for i in range(10)}, gr.BarPlot(visible=False)
        return predict_sketch(img)
    
    submit_btn.click(
        fn=predict_sketch,
        inputs=sketch,
        outputs=[result_label, result_bar]
    )
    
    clear_btn.click(
        fn=lambda: (None, {str(i): 0 for i in range(10)}, gr.BarPlot(visible=False)),
        inputs=None,
        outputs=[sketch, result_label, result_bar]
    )
    
    gr.Markdown("---")
    gr.Markdown("### Instructions:")
    gr.Markdown("1. Draw a digit (0-9) on the canvas")
    gr.Markdown("2. Click 'Submit' to recognize")
    gr.Markdown("3. Click 'Clear' to draw again")
    gr.Markdown("4. The bar chart shows Top-3 prediction probabilities")

demo.launch()
