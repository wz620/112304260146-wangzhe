import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download


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


def load_model():
    model = DigitCNN()
    try:
        local_model_path = os.path.join(os.path.dirname(__file__), "..", "data", "model.pth")
        if os.path.exists(local_model_path):
            model.load_state_dict(torch.load(local_model_path, map_location=torch.device('cpu')))
        else:
            model_path = hf_hub_download(repo_id="qd1234/wz-112304260146", filename="model.pth")
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"警告: 无法加载模型 ({e}), 使用随机初始化的模型")
    model.eval()
    return model


model = load_model()


def preprocess_image(img: Image.Image) -> torch.Tensor:
    if img is None:
        return None
    if img.mode != 'L':
        img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).reshape(1, 1, 28, 28)
    return img_tensor


def create_probability_plot(probs):
    fig, ax = plt.subplots(figsize=(6, 4))
    digits = list(range(10))
    ax.bar(digits, probs, color='steelblue', alpha=0.7)
    ax.set_xlabel('Digit')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    ax.set_xticks(digits)
    return fig


def predict(img: Image.Image, history: list) -> tuple:
    if img is None:
        return "0", "0%", plt.figure(), [], history
    
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
    
    plot_fig = create_probability_plot(probs[0].tolist())
    
    top3_table = [[i+1, labels[i], f"{values[i]:.1f}%"] for i in range(3)]
    
    history.insert(0, [str(pred), f"{confidence*100:.1f}%", ", ".join(labels)])
    if len(history) > 10:
        history = history[:10]
    
    return str(pred), f"{confidence*100:.2f}%", plot_fig, top3_table, history


with gr.Blocks(title="手写数字识别系统") as demo:
    gr.Markdown("# 手写数字识别系统")
    gr.Markdown("基于 CNN 的在线识别页面，支持上传图片、网页手写板、Top-3 预测、概率分布和连续识别历史。")
    
    history_state = gr.State([])
    
    with gr.Tab("上传图片识别"):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="上传手写数字图片", height=300)
                with gr.Row():
                    clear_btn1 = gr.Button("清空", size="sm")
                    submit_btn1 = gr.Button("识别", variant="primary", size="sm")
            with gr.Column(scale=1):
                with gr.Row():
                    pred_output = gr.Textbox(label="预测结果", scale=1)
                    conf_output = gr.Textbox(label="置信度", scale=1)
                bar_output = gr.Plot(label="概率分布")
        gr.Markdown("### Top-3 预测")
        top3_output = gr.Dataframe(
            headers=["Rank", "Digit", "Probability"],
            label="Top-3 结果"
        )
        submit_btn1.click(
            fn=predict,
            inputs=[image_input, history_state],
            outputs=[pred_output, conf_output, bar_output, top3_output, history_state]
        )
        clear_btn1.click(fn=lambda: (None, "0", plt.figure(), [], []), outputs=[image_input, pred_output, bar_output, top3_output, history_state])
    
    with gr.Tab("在线手写板识别"):
        with gr.Row():
            with gr.Column(scale=1):
                sketch = gr.Sketchpad(
                    label="手写板",
                    type="pil",
                    canvas_size=(280, 280),
                )
                with gr.Row():
                    clear_btn2 = gr.Button("清空", size="sm")
                    submit_btn2 = gr.Button("识别", variant="primary", size="sm")
            with gr.Column(scale=1):
                with gr.Row():
                    pred_output2 = gr.Textbox(label="预测结果", scale=1)
                    conf_output2 = gr.Textbox(label="置信度", scale=1)
                bar_output2 = gr.Plot(label="概率分布")
        gr.Markdown("### Top-3 预测")
        top3_output2 = gr.Dataframe(
            headers=["Rank", "Digit", "Probability"],
            label="Top-3 结果"
        )
        
        submit_btn2.click(
            fn=predict,
            inputs=[sketch, history_state],
            outputs=[pred_output2, conf_output2, bar_output2, top3_output2, history_state]
        )
        clear_btn2.click(fn=lambda: (None, "0", plt.figure(), [], []), outputs=[sketch, pred_output2, bar_output2, top3_output2, history_state])
    
    gr.Markdown("### 连续识别历史")
    with gr.Row():
        clear_history_btn = gr.Button("清空历史", size="sm")
        history_output = gr.Dataframe(
            headers=["预测", "置信度", "Top-3"],
            label="识别历史"
        )
    clear_history_btn.click(fn=lambda: [], outputs=history_output)

demo.launch()
