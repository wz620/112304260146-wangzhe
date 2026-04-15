# 机器学习实验：基于 TF-IDF/Word2Vec 的情感预测

## 1. 学生信息
- **姓名**：王哲
- **学号**：112304260146
- **班级**：数据1231

---

## 2. 实验任务
本实验基于给定文本数据，分别使用 **TF-IDF** 和 **Word2Vec** 将文本转为向量特征，结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理（保留否定词）
- TF-IDF / Word2Vec 特征提取
- N-gram 短语模式
- 分类模型训练与对比
- Kaggle 结果提交与分析

---

## 3. 比赛与提交信息
- **比赛名称**：Bag of Words Meets Bags of Popcorn
- **比赛链接**：https://www.kaggle.com/competitions/word2vec-nlp-tutorial/data
- **提交日期**：2026-04-15
- **GitHub 仓库地址**：https://github.com/wz620/112304260146-wangzhe

> 注意：GitHub 仓库首页或 README 页面中，必须能看到"姓名 + 学号"，否则无效。

---

## 4. Kaggle 成绩
- **Public Score**：0.96753
- **Private Score**（如有）：
- **排名**（如能看到可填写）：

---

## 5. Kaggle 截图
![Kaggle截图](./images/kaggle.png)

---

## 6. 实验方法说明

### （1）文本预处理
- HTML 标签去除
- 缩写词展开（n't → not, 're → are 等）
- 分词
- 去停用词（保留否定词）
- 转小写

**关键设计**：
```python
# 保留否定词
NEGATION_WORDS = {"no", "not", "nor", "never", "none", "n't"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS
```

这确保了 "not good" 这样的否定表达不会被破坏，对情感分析至关重要。

---

### （2）特征提取

本实验实现了两种特征提取方式：

#### TF-IDF 特征
- N-gram 范围：(1, 4) - 1元到4元语法
- 最大特征数：100,000
- 最小文档频率：2
- 最大文档频率：0.95
- TF 归一化：sublinear_tf=True

#### Word2Vec 特征
- 词向量维度：300
- 上下文窗口：10
- 最小词频：40
- 训练算法：Skip-gram + Hierarchical Softmax
- 句子向量：平均词向量 / 词聚类特征

---

### （3）分类模型
测试了以下模型：
- Logistic Regression
- LinearSVC
- Random Forest

---

## 7. 实验结果

### 最佳模型配置
- **特征提取**：TF-IDF (1-4 gram, 100,000 features)
- **分类模型**：LinearSVC (C=0.5)
- **CV AUC**：**0.96835**

### 各折得分

| Fold | ROC-AUC |
|------|---------|
| 1 | 0.96605 |
| 2 | 0.96653 |
| 3 | 0.97086 |
| 4 | 0.96834 |
| 5 | 0.96997 |
| **Mean** | **0.96835** |

---

## 8. 运行代码

```bash
cd code
python experiment_word2vec_auc.py \
    --feature tfidf \
    --classifier linear_svc \
    --lr-c 0.5 \
    --ngram-min 1 \
    --ngram-max 4 \
    --max-features 100000
```

---

## 9. 文件说明

```
baomihua/
├─ code/
│  └─ experiment_word2vec_auc.py    # 主代码
├─ data/
│  ├─ labeledTrainData.tsv/        # 训练数据
│  ├─ unlabeledTrainData.tsv/      # 无标签数据
│  ├─ testData.tsv/               # 测试数据
│  ├─ DATASET.md                  # 数据集说明
│  └─ sampleSubmission.csv        # 提交样例
├─ images/
│  └─ kaggle.png                  # Kaggle 截图
├─ logs/
│  └─ attempt_log.csv             # 实验日志
├─ report/
│  └─ experiment_report.md        # 实验报告
├─ submission/
│  └─ submission_*.csv           # 提交文件
├─ .gitignore
└─ README.md
```

---

## 10. 实验总结

本次实验成功完成了情感分析任务，最终使用 **TF-IDF (1-4 gram) + LinearSVC** 取得了 **0.96835** 的交叉验证 AUC。

主要收获：
1. **保留否定词**：对情感分析效果有显著提升
2. **N-gram 短语模式**：可以捕获丰富的情感表达
3. **简单模型同样有效**：LinearSVC 在文本分类任务上表现优异
4. **稀疏矩阵优化**：可以处理大规模特征
