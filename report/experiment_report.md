# 情感分析实验报告

## 实验信息

- **实验名称**：基于 TF-IDF 和逻辑回归的情感分析
- **实验日期**：2026-04-15
- **学生姓名**：王哲
- **学号**：112304260146
- **班级**：数据1231

---

## 1. 实验目标

完成 Kaggle 竞赛 "Bag of Words Meets Bags of Popcorn" 的情感预测任务，使用 TF-IDF 特征提取结合简单分类模型（逻辑回归、LinearSVC），优化 AUC 评分。

---

## 2. 数据集说明

| 数据集 | 样本数 | 说明 |
|--------|--------|------|
| labeledTrainData.tsv | 25,000 | 带标签的训练数据 |
| unlabeledTrainData.tsv | 50,000 | 无标签数据（用于 Word2Vec） |
| testData.tsv | 25,000 | 测试集（需预测） |

---

## 3. 实验方法

### 3.1 文本预处理

1. **HTML 标签去除**：使用 BeautifulSoup 去除 `<br />` 等标签
2. **缩写词展开**：将 `n't` 展开为 `not`，`'re` 展开为 `are` 等
3. **转小写**：统一转为小写
4. **去停用词**：移除英文停用词，但**保留否定词**（no, not, nor, never, none, n't）
5. **正则分词**：提取英文单词

### 3.2 特征提取

使用 **TF-IDF** 进行特征提取，关键参数：
- **N-gram 范围**：(1, 4) - 1元到4元语法
- **最大特征数**：100,000
- **最小文档频率**：2
- **最大文档频率**：0.95
- **TF 归一化**：sublinear_tf=True

### 3.3 分类模型

测试了以下模型：
1. **Logistic Regression**
2. **LinearSVC**

---

## 4. 实验结果

### 4.1 交叉验证结果

| 模型 | N-gram | Max Features | CV AUC | Std |
|------|--------|--------------|-------|-----|
| LogisticRegression | 1-3 | 50,000 | 0.96305 | 0.00205 |
| **LinearSVC** | **1-4** | **100,000** | **0.96835** | **0.00187** |

### 4.2 最优模型配置

- **特征提取**：TF-IDF (1-4 gram, 100,000 features)
- **分类模型**：LinearSVC (C=0.5)
- **CV AUC**：0.96835

### 4.3 提交文件

| 文件名 | 说明 |
|--------|------|
| submission_2026-04-15T21-31-04..._linear_svc_c0_5_tfidf_ngram1_4_max100000.csv | 最终提交文件 |

---

## 5. 实验分析

### 5.1 为什么使用 TF-IDF 而非 Word2Vec

1. **训练速度**：TF-IDF 特征提取速度远快于 Word2Vec
2. **效果更好**：在本数据集上，TF-IDF 的 AUC (0.968) 显著高于 Word2Vec (约 0.95)
3. **资源消耗**：TF-IDF 不需要大量内存存储词向量

### 5.2 保留否定词的重要性

在预处理时特意保留否定词（not, no, never 等），因为：
- 否定词对情感分析至关重要
- "not good" 和 "good" 情感完全相反
- 移除否定词会严重损害模型性能

### 5.3 N-gram 短语模式

使用 (1,4) 的 N-gram 范围可以捕获：
- **unigram**：单词级别特征
- **bigram**：短语如 "not good", "very bad"
- **trigram**：更长的情感短语
- **4-gram**：捕捉更复杂的表达

---

## 6. 代码说明

### 6.1 核心参数

```bash
python experiment_word2vec_auc.py \
    --feature tfidf \
    --classifier linear_svc \
    --lr-c 0.5 \
    --ngram-min 1 \
    --ngram-max 4 \
    --max-features 100000
```

### 6.2 关键优化点

1. **使用稀疏矩阵**：避免将 TF-IDF 矩阵转为密集矩阵，节省内存
2. **Sigmoid 转换**：将 LinearSVC 的决策函数转为概率值
3. **正确的 CSV 格式**：严格按照 sampleSubmission.csv 格式输出

---

## 7. 文件结构

```
baomihua/
├─ code/
│  └─ experiment_word2vec_auc.py    # 主代码
├─ data/
│  ├─ labeledTrainData.tsv/         # 训练数据
│  ├─ unlabeledTrainData.tsv/       # 无标签数据
│  ├─ testData.tsv/                 # 测试数据
│  ├─ DATASET.md                    # 数据集说明
│  └─ sampleSubmission.csv          # 提交样例
├─ images/
│  └─ kaggle.png                    # Kaggle 截图
├─ logs/
│  └─ attempt_log.csv               # 实验日志
├─ report/
│  └─ experiment_report.md          # 本报告
├─ results/
│  └─ submission/                   # 提交文件
│     └─ submission_*.csv
├─ .gitignore
└─ README.md
```

---

## 8. 总结

本次实验成功完成了情感分析任务，最终使用 **TF-IDF (1-4 gram) + LinearSVC** 取得了 **0.96835** 的交叉验证 AUC。

关键收获：
1. 否定词对情感分析非常重要，预处理时必须保留
2. N-gram 短语模式可以显著提升效果
3. 简单模型（LinearSVC）在文本分类任务上表现优异
4. 使用稀疏矩阵可以处理大规模特征

---

## 9. 参考

- Kaggle 竞赛：https://www.kaggle.com/competitions/word2vec-nlp-tutorial
- scikit-learn 文档：https://scikit-learn.org/
- Gensim Word2Vec：https://radimrehurek.com/gensim/models/word2vec.html
