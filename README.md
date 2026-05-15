# 交通标志目标检测

本项目使用 YOLOv8 进行交通标志目标检测任务。

## 项目结构

```
路标/
├── code/                   # 实验代码
│   ├── train_model.py      # 训练脚本
│   ├── run_infer.py        # 推理脚本
│   ├── baseline_infer.py   # 基准推理脚本
│   └── data.yaml           # 数据配置文件
├── report/                 # 实验报告
│   └── 第四次实验报告.md   # 实验报告
├── results/                # 实验结果
│   ├── results.csv         # 训练结果数据
│   ├── results.png         # 训练结果图表
│   ├── BoxPR_curve.png     # PR曲线
│   └── confusion_matrix.png # 混淆矩阵
└── README.md               # 本文件
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- Ultralytics 8.0+
- CUDA 12.1+ (GPU版本)

## 安装依赖

```bash
pip install ultralytics pandas pillow
```

## 训练模型

```bash
cd 路标
python code/train_model.py
```

## 运行推理

```bash
cd 路标
python code/run_infer.py
```

## 学生信息

- **姓名**：王哲
- **学号**：112304260146
- **班级**：数据1231
