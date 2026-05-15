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

## 类别说明

| 类别ID | 名称 |
|--------|------|
| 0 | Green Light |
| 1 | Red Light |
| 2 | Speed Limit 10 |
| 3 | Speed Limit 100 |
| 4 | Speed Limit 110 |
| 5 | Speed Limit 120 |
| 6 | Speed Limit 20 |
| 7 | Speed Limit 30 |
| 8 | Speed Limit 40 |
| 9 | Speed Limit 50 |
| 10 | Speed Limit 60 |
| 11 | Speed Limit 70 |
| 12 | Speed Limit 80 |
| 13 | Speed Limit 90 |
| 14 | Stop |

## 训练结果

- 模型: YOLOv8m
- Epochs: 50
- mAP50: 87%
- Recall: 81%
