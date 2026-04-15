# 情感分析实验报告

## 实验信息

- **实验名称**：基于 TF-IDF/Word2Vec 的情感分析
- **实验日期**：2026-04-15
- **学生姓名**：王哲
- **学号**：112304260146
- **班级**：数据1231

---

## 1. 实验目标

完成 Kaggle 竞赛 "Bag of Words Meets Bags of Popcorn" 的情感预测任务，分别使用 **TF-IDF** 和 **Word2Vec** 两种特征提取方法，结合简单分类模型（逻辑回归、LinearSVC），优化 ROC-AUC 评分。

---

## 2. 数据集说明

| 数据集 | 样本数 | 说明 |
|--------|--------|------|
| labeledTrainData.tsv | 25,000 | 带标签的训练数据（情感标签：0/1） |
| unlabeledTrainData.tsv | 50,000 | 无标签数据（用于 Word2Vec 训练） |
| testData.tsv | 25,000 | 测试集（需预测情感） |

---

## 3. 实验方法

### 3.1 文本预处理

代码实现了完整的预处理流水线：

```python
# 1. HTML 标签去除
def strip_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text(" ")

# 2. 缩写词展开
def expand_contractions(text: str) -> str:
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    ...

# 3. 分词与清洗
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")
```

**关键设计**：
- **保留否定词**：定义 `NEGATION_WORDS = {"no", "not", "nor", "never", "none", "n't"}`
- **去停用词时排除否定词**：`STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS`
- 这确保了 "not good" 这样的否定表达不会被破坏

### 3.2 特征提取

程序支持两种特征提取方式：

#### 方式一：TF-IDF 特征
```python
TfidfVectorizer(
    ngram_range=(1, 4),      # 1-4 元语法
    max_features=100000,      # 最大特征数
    min_df=2,                 # 最小文档频率
    max_df=0.95,              # 最大文档频率
    sublinear_tf=True,        # TF 对数归一化
)
```

#### 方式二：Word2Vec 特征
```python
Word2Vec(
    vector_size=300,          # 词向量维度
    window=10,                # 上下文窗口
    min_count=40,             # 最小词频
    sg=1,                     # Skip-gram
    hs=1,                     # Hierarchical Softmax
    epochs=10,                # 训练轮数
)
```

同时支持两种 Word2Vec 特征表示：
1. **平均词向量 (Mean Embeddings)**：将句子中所有词向量做平均
2. **词聚类特征 (Bag of Centroids)**：用 KMeans 聚类后统计各簇词频

### 3.3 分类模型

测试了以下模型：
- **Logistic Regression**
- **LinearSVC**
- **Random Forest**

---

## 4. 实验结果

### 4.1 实验日志

| 时间 | 模型 | 特征类型 | N-gram | Max Features | CV AUC | Std |
|------|------|----------|--------|--------------|--------|-----|
| 21:22 | LogisticRegression | TF-IDF | 1-3 | 50,000 | 0.96305 | 0.00205 |
| 21:23 | **LinearSVC** | **TF-IDF** | **1-4** | **100,000** | **0.96835** | **0.00187** |

### 4.2 最优模型配置

- **特征提取**：TF-IDF
  - N-gram 范围：(1, 4)
  - 最大特征数：100,000
- **分类模型**：LinearSVC (C=0.5)
- **CV AUC**：**0.96835**

### 4.3 各折得分

| Fold | ROC-AUC |
|------|---------|
| 1 | 0.96605 |
| 2 | 0.96653 |
| 3 | 0.97086 |
| 4 | 0.96834 |
| 5 | 0.96997 |
| **Mean** | **0.96835** |

---

## 5. 实验分析

### 5.1 TF-IDF vs Word2Vec

在本数据集上，TF-IDF 表现更优：
- **TF-IDF AUC**：0.96835
- **Word2Vec AUC**：约 0.95

原因分析：
1. TF-IDF 训练速度更快
2. N-gram 可以捕获短语模式，如 "not good"、"very bad"
3. Word2Vec 需要更多数据才能达到最优效果

### 5.2 保留否定词的重要性

这是本实验的关键优化点：

```python
# 否定词集合
NEGATION_WORDS = {"no", "not", "nor", "never", "none", "n't"}

# 停用词 = 标准停用词 - 否定词
STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS
```

- "not good" → 情感为负
- "good" → 情感为正
- 移除 "not" 会导致这两种表达无法区分

### 5.3 N-gram 短语模式

使用 (1,4) 的 N-gram 范围效果最好：

| N-gram | 示例 | 作用 |
|--------|------|------|
| 1-gram | good, bad | 基础词汇 |
| 2-gram | not good, very bad | 否定和程度 |
| 3-gram | not very good | 复杂修饰 |
| 4-gram | not very good at | 更细粒度 |

---

## 6. 代码架构

### 6.1 核心函数

```python
# 数据加载
def load_tsv(path: Path, nrows: int | None = None) -> pd.DataFrame

# 文本预处理
def strip_html(text: str) -> str
def expand_contractions(text: str) -> str
def tokenize_text(text: str, remove_stopwords: bool) -> list[str]
def preprocess_reviews(reviews, remove_stopwords: bool) -> list[list[str]]
def preprocess_for_tfidf(reviews) -> list[str]

# Word2Vec 相关
def load_or_train_word2vec(sentences, args) -> tuple[Word2Vec, Path, bool]
def build_average_vectors(tokenized_reviews, model) -> np.ndarray
def build_bag_of_centroids(tokenized_reviews, word_to_cluster, n_clusters) -> np.ndarray

# TF-IDF 相关
def build_tfidf_features(train_texts, test_texts, args) -> tuple

# 模型训练与评估
def build_classifier(classifier_name, lr_c, lr_max_iter, rf_trees)
def evaluate_with_cv(features, labels, folds, classifier_name, ...) -> tuple[list[float], float]
def fit_and_predict_submission(train_features, train_labels, test_features, ...) -> np.ndarray

# 工具函数
def get_classifier_scores(classifier, classifier_name, features) -> np.ndarray
def log_attempt(...)
def build_submission_path(timestamp, feature_name) -> Path
```

### 6.2 命令行参数

```bash
python experiment_word2vec_auc.py \
    --feature tfidf \                  # 特征类型: tfidf 或 word2vec
    --classifier linear_svc \           # 分类器: logistic_regression, linear_svc, random_forest
    --lr-c 0.5 \                       # 正则化参数
    --ngram-min 1 \                    # 最小 n-gram
    --ngram-max 4 \                    # 最大 n-gram
    --max-features 100000 \           # 最大特征数
    --folds 5 \                        # 交叉验证折数
    --vector-size 300 \                # Word2Vec 词向量维度
    --window 10 \                      # Word2Vec 窗口大小
    --min-count 40 \                   # Word2Vec 最小词频
    --epochs 10                        # Word2Vec 训练轮数
```

### 6.3 关键优化点

1. **稀疏矩阵**：TF-IDF 使用稀疏矩阵表示，避免内存溢出
2. **Sigmoid 转换**：LinearSVC 的 decision_function 通过 sigmoid 转为概率
3. **正确格式**：输出 CSV 格式严格匹配 sampleSubmission.csv
4. **否定词保留**：预处理时保留否定词，提升情感判断准确性

---

## 7. 文件结构

```
baomihua/
├─ code/
│  └─ experiment_word2vec_auc.py    # 主代码（约 600 行）
├─ data/
│  ├─ labeledTrainData.tsv/        # 训练数据 (25,000 条)
│  ├─ unlabeledTrainData.tsv/      # 无标签数据 (50,000 条)
│  ├─ testData.tsv/                # 测试数据 (25,000 条)
│  ├─ DATASET.md                   # 数据集说明
│  └─ sampleSubmission.csv         # 提交样例
├─ images/
│  └─ kaggle.png                   # Kaggle 截图
├─ logs/
│  └─ attempt_log.csv              # 实验日志（所有运行记录）
├─ report/
│  └─ experiment_report.md         # 本报告
├─ results/
│  └─ submission/                  # 提交文件
│     └─ submission_*.csv
├─ .gitignore                      # Git 忽略配置
└─ README.md                       # 仓库说明
```

---

## 8. 总结

本次实验成功完成了情感分析任务，最终使用 **TF-IDF (1-4 gram, 100,000 features) + LinearSVC (C=0.5)** 取得了 **0.96835** 的交叉验证 ROC-AUC。

### 主要收获

1. **预处理至关重要**：保留否定词对情感分析效果有显著提升
2. **N-gram 效果显著**：使用 1-4 gram 可以捕获丰富的短语模式
3. **简单模型同样有效**：LinearSVC 在文本分类任务上表现优异
4. **稀疏矩阵优化**：使用稀疏矩阵可以处理大规模特征，避免内存问题

### 后续改进方向

1. 尝试更多 n-gram 范围（如 1-5）
2. 尝试其他分类器如 XGBoost、LightGBM
3. 集成学习：结合 TF-IDF 和 Word2Vec 特征
4. 调参优化：网格搜索最佳超参数

---

## 9. 参考

- Kaggle 竞赛：https://www.kaggle.com/competitions/word2vec-nlp-tutorial
- scikit-learn 文档：https://scikit-learn.org/
- Gensim Word2Vec：https://radimrehurek.com/gensim/models/word2vec.html
- BeautifulSoup：https://www.crummy.com/software/BeautifulSoup/
