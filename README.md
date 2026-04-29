# 机器学习实验：基于 CNN 的手写数字识别

## 1. 学生信息
- **姓名**：王哲
- **学号**：112304260146
- **班级**：数据1231

---

## 2. 实验任务
本实验基于 MNIST 手写数字数据集，使用 **卷积神经网络（CNN）** 完成从模型训练到应用部署的完整流程。

本实验重点包括：
- CNN 模型搭建与训练
- 超参数调优
- 数据增强
- Web 应用部署（Gradio）
- 交互式手写识别系统

---

## 3. 比赛与提交信息
- **比赛名称**：Digit Recognizer
- **比赛链接**：https://www.kaggle.com/competitions/digit-recognizer
- **GitHub 仓库地址**：https://github.com/wz620/112304260146-wangzhe
- **HuggingFace Space**：https://huggingface.co/spaces/qd1234/wz-112304260146

> 注意：GitHub 仓库首页或 README 页面中，必须能看到"姓名 + 学号"，否则无效。

---

## 4. Kaggle 成绩
- **Public Score**：0.98514

---

## 5. 实验方法说明

### （1）模型结构
采用三层卷积神经网络结构：

```
输入(1×28×28) 
→ Conv(1→32) + BN + ReLU + MaxPool + Dropout(0.1)
→ Conv(32→64) + BN + ReLU + MaxPool + Dropout(0.2)  
→ Conv(64→128) + BN + ReLU + MaxPool + Dropout(0.3)
→ Flatten → Linear(256) → Dropout(0.4) → Linear(10)
→ 输出(10类)
```

### （2）训练配置
- **优化器**：AdamW (lr=1e-3, weight_decay=1e-4)
- **学习率调度**：CosineAnnealingLR (T_max=8)
- **Batch Size**：512
- **Early Stopping**：patience=4
- **验证集准确率**：99.37%

---

## 6. 文件说明

```
digit-recognizer/
├─ code/
│  ├─ app.py                    # 实验二：图片上传预测 Web 应用
│  ├─ app_sketch.py             # 实验三：手写板交互系统
│  ├─ train_save_model.py       # 模型训练脚本
│  ├─ train_and_plot.py         # 训练并绘制 Loss 曲线
│  ├─ train_and_submit.py       # 训练并提交 Kaggle
│  ├─ dnn_train.py              # DNN 训练代码
│  ├─ upload_hf.py              # 上传 HuggingFace 脚本
│  └─ upload.bat                # 上传脚本
├─ data/
│  ├─ train.csv                 # 训练数据（需从 Kaggle 下载）
│  ├─ test.csv                  # 测试数据（需从 Kaggle 下载）
│  ├─ submission.csv            # 提交文件
│  ├─ model.pth                 # 训练好的模型权重
│  ├─ requirements.txt          # Python 依赖
│  └─ DATASET.md                # 数据集说明
├─ images/
│  └─ loss_curve.png            # Loss 曲线图
├─ report/
│  └─ CNN手写数字识别实验模板.md # 实验报告模板
├─ .gitignore
└─ README.md
```

---

## 7. 本地运行

### 安装依赖

```bash
cd digit-recognizer/code
pip install -r ../data/requirements.txt
```

### 运行实验二（图片上传）

```bash
python app.py
```

### 运行实验三（手写板）

```bash
python app_sketch.py
```

---

## 8. 数据说明

请从 Kaggle 下载数据：
- [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data)

需要下载的文件：
- train.csv
- test.csv

---

## 9. 实验总结

本次实验成功完成了手写数字识别任务，最终使用 **CNN + AdamW** 取得了 **0.98514** 的 Kaggle 评分。
