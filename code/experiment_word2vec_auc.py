from __future__ import annotations

import argparse
from datetime import datetime
import re
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC


DATA_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = DATA_DIR / "data"
ATTEMPT_LOG_PATH = DATA_DIR / "logs" / "attempt_log.csv"
SUBMISSION_DIR = DATA_DIR / "submission"

NEGATION_WORDS = {"no", "not", "nor", "never", "none", "n't"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a sentiment classifier with Word2Vec or TF-IDF features."
    )
    parser.add_argument(
        "--feature",
        choices=["word2vec", "tfidf"],
        default="tfidf",
        help="Feature extraction method: Word2Vec or TF-IDF with n-grams.",
    )
    parser.add_argument(
        "--classifier",
        choices=["logistic_regression", "random_forest", "linear_svc"],
        default="logistic_regression",
        help="Classifier used on top of features.",
    )
    parser.add_argument("--vector-size", type=int, default=300, help="Word2Vec embedding size.")
    parser.add_argument("--window", type=int, default=10, help="Word2Vec context window size.")
    parser.add_argument("--min-count", type=int, default=40, help="Minimum token frequency.")
    parser.add_argument("--epochs", type=int, default=10, help="Word2Vec training epochs.")
    parser.add_argument("--folds", type=int, default=5, help="Stratified CV fold count.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker count.")
    parser.add_argument(
        "--downsample",
        type=float,
        default=1e-3,
        help="Word2Vec frequent-word downsampling threshold.",
    )
    parser.add_argument(
        "--kmeans-clusters",
        type=int,
        default=10,
        help="Number of KMeans clusters for bag-of-centroids features.",
    )
    parser.add_argument(
        "--lr-c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for LogisticRegression.",
    )
    parser.add_argument(
        "--lr-max-iter",
        type=int,
        default=2000,
        help="Maximum iterations for LogisticRegression.",
    )
    parser.add_argument(
        "--rf-trees",
        type=int,
        default=100,
        help="Number of trees for RandomForestClassifier.",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=1,
        help="Minimum n-gram size for TF-IDF.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=4,
        help="Maximum n-gram size for TF-IDF.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=150000,
        help="Maximum number of TF-IDF features.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap for labeled rows, useful for debugging.",
    )
    parser.add_argument(
        "--max-unlabeled-rows",
        type=int,
        default=None,
        help="Optional cap for unlabeled rows used in Word2Vec training.",
    )
    parser.add_argument(
        "--force-retrain-w2v",
        action="store_true",
        help="Retrain Word2Vec even if a cached model exists.",
    )
    return parser.parse_args()


def load_tsv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', header=0, quoting=3, nrows=nrows)
    df['id'] = df['id'].astype(str).str.strip('"')
    return df


def normalize_tsv_field(value: str) -> str:
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    return value.replace('""', '"')


def strip_html(text: str) -> str:
    if "<" in text and ">" in text:
        return BeautifulSoup(text, "html.parser").get_text(" ")
    return text


def expand_contractions(text: str) -> str:
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    return text


def tokenize_text(text: str, *, remove_stopwords: bool) -> list[str]:
    text = expand_contractions(text)
    cleaned = strip_html(text).lower()
    tokens = TOKEN_PATTERN.findall(cleaned)
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOP_WORDS]
    return tokens


def preprocess_reviews(reviews: Iterable[str], *, remove_stopwords: bool) -> list[list[str]]:
    return [tokenize_text(review, remove_stopwords=remove_stopwords) for review in reviews]


def preprocess_for_tfidf(reviews: Iterable[str]) -> list[str]:
    processed = []
    for review in reviews:
        review = expand_contractions(str(review))
        review = strip_html(review)
        review = re.sub(r"[^a-zA-Z]", " ", review.lower())
        processed.append(review)
    return processed


def get_w2v_cache_path(args: argparse.Namespace) -> Path:
    cache_name = (
        "word2vec_"
        f"vs{args.vector_size}_w{args.window}_mc{args.min_count}_"
        f"ep{args.epochs}_ds{args.downsample:g}_sg1_hs1.model"
    )
    return DATA_DIR / cache_name


def load_or_train_word2vec(
    sentences: list[list[str]],
    args: argparse.Namespace,
) -> tuple[Word2Vec, Path, bool]:
    cache_path = get_w2v_cache_path(args)
    if cache_path.exists() and not args.force_retrain_w2v:
        return Word2Vec.load(str(cache_path)), cache_path, False

    model = Word2Vec(
        sentences=sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=1,
        hs=1,
        negative=0,
        sample=args.downsample,
        epochs=args.epochs,
        seed=42,
    )
    model.save(str(cache_path))
    return model, cache_path, True


def build_average_vectors(tokenized_reviews: list[list[str]], model: Word2Vec) -> np.ndarray:
    vector_size = model.wv.vector_size
    features = np.zeros((len(tokenized_reviews), vector_size), dtype=np.float32)

    for idx, tokens in enumerate(tokenized_reviews):
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if vectors:
            features[idx] = np.mean(vectors, axis=0)
    return features


def fit_kmeans_on_word_vectors(model: Word2Vec, n_clusters: int) -> tuple[KMeans, list[str], np.ndarray]:
    vocab_words = list(model.wv.index_to_key)
    vocab_vectors = model.wv[vocab_words]
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(vocab_vectors)
    return kmeans, vocab_words, cluster_labels


def build_bag_of_centroids(
    tokenized_reviews: list[list[str]],
    word_to_cluster: dict[str, int],
    n_clusters: int,
) -> np.ndarray:
    features = np.zeros((len(tokenized_reviews), n_clusters), dtype=np.float32)
    for row_idx, tokens in enumerate(tokenized_reviews):
        for token in tokens:
            cluster_id = word_to_cluster.get(token)
            if cluster_id is not None:
                features[row_idx, cluster_id] += 1.0
    return features


def build_tfidf_features(
    train_texts: list[str],
    test_texts: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        token_pattern=r'\b\w+\b',
    )
    
    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)
    
    return train_features, test_features, vectorizer


def evaluate_with_cv(
    features,
    labels: np.ndarray,
    folds: int,
    classifier_name: str,
    lr_c: float,
    lr_max_iter: int,
    rf_trees: int,
) -> tuple[list[float], float]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores: list[float] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(features, labels), start=1):
        classifier = build_classifier(classifier_name, lr_c, lr_max_iter, rf_trees)
        classifier.fit(features[train_idx], labels[train_idx])
        valid_scores = get_classifier_scores(classifier, classifier_name, features[valid_idx])
        auc = roc_auc_score(labels[valid_idx], valid_scores)
        scores.append(auc)
        print(f"Fold {fold_idx}: ROC-AUC = {auc:.5f}")

    return scores, float(np.mean(scores))


def upsert_attempt_log(row: dict[str, object]) -> None:
    ATTEMPT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_row_df = pd.DataFrame([row])
    if ATTEMPT_LOG_PATH.exists():
        existing_df = pd.read_csv(ATTEMPT_LOG_PATH)
        for column in new_row_df.columns:
            if column not in existing_df.columns:
                existing_df[column] = ""
        for column in existing_df.columns:
            if column not in new_row_df.columns:
                new_row_df[column] = ""
        combined_df = pd.concat([existing_df, new_row_df[existing_df.columns]], ignore_index=True)
    else:
        combined_df = new_row_df
    combined_df.to_csv(ATTEMPT_LOG_PATH, index=False)


def fit_and_predict_submission(
    train_features,
    train_labels: np.ndarray,
    test_features,
    classifier_name: str,
    lr_c: float,
    lr_max_iter: int,
    rf_trees: int,
) -> np.ndarray:
    classifier = build_classifier(classifier_name, lr_c, lr_max_iter, rf_trees)
    classifier.fit(train_features, train_labels)
    return get_classifier_scores(classifier, classifier_name, test_features)


def build_classifier(classifier_name: str, lr_c: float, lr_max_iter: int, rf_trees: int):
    if classifier_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=rf_trees,
            random_state=42,
            n_jobs=-1,
        )

    if classifier_name == "linear_svc":
        return LinearSVC(C=lr_c, max_iter=lr_max_iter, random_state=42)

    return LogisticRegression(
        C=lr_c,
        max_iter=lr_max_iter,
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    )


def get_classifier_scores(classifier, classifier_name: str, features: np.ndarray) -> np.ndarray:
    if classifier_name == "linear_svc":
        scores = classifier.decision_function(features)
        return 1 / (1 + np.exp(-scores))
    return classifier.predict_proba(features)[:, 1]


def log_attempt(
    *,
    timestamp: str,
    feature_type: str,
    scores: list[float],
    mean_auc: float,
    args: argparse.Namespace,
    labeled_rows: int,
    unlabeled_rows: int,
    w2v_model_path: Path | None = None,
    submission_path: Path | None = None,
) -> None:
    roc_auc_std = "" if not scores else round(float(np.std(scores)), 6)
    fold_scores = "" if not scores else ";".join(f"{score:.6f}" for score in scores)
    upsert_attempt_log(
        {
            "timestamp": timestamp,
            "model_type": args.classifier,
            "feature_type": feature_type,
            "roc_auc_mean": round(mean_auc, 6),
            "roc_auc_std": roc_auc_std,
            "fold_scores": fold_scores,
            "vector_size": args.vector_size,
            "window": args.window,
            "min_count": args.min_count,
            "epochs": args.epochs,
            "folds": args.folds,
            "workers": args.workers,
            "downsample": args.downsample,
            "kmeans_clusters": args.kmeans_clusters,
            "lr_c": args.lr_c,
            "lr_max_iter": args.lr_max_iter,
            "rf_trees": args.rf_trees,
            "labeled_rows": labeled_rows,
            "unlabeled_rows": unlabeled_rows,
            "w2v_model_path": str(w2v_model_path) if w2v_model_path else "",
            "submission_path": "" if submission_path is None else str(submission_path),
            "ngram_range": f"{args.ngram_min}-{args.ngram_max}",
            "max_features": args.max_features,
        }
    )


def build_submission_path(timestamp: str, feature_name: str) -> Path:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    safe_timestamp = timestamp.replace(":", "-")
    safe_feature_name = re.sub(r"[^a-zA-Z0-9_]+", "_", feature_name)
    filename = f"submission_{safe_timestamp}_{safe_feature_name}.csv"
    return SUBMISSION_DIR / filename


def main() -> None:
    args = parse_args()

    labeled_path = DATA_PATH / "labeledTrainData.tsv" / "labeledTrainData.tsv"
    unlabeled_path = DATA_PATH / "unlabeledTrainData.tsv" / "unlabeledTrainData.tsv"
    test_path = DATA_PATH / "testData.tsv" / "testData.tsv"
    
    print("Loading data...")
    labeled_df = load_tsv(labeled_path, nrows=args.max_train_rows)
    unlabeled_df = load_tsv(unlabeled_path, nrows=args.max_unlabeled_rows)
    test_df = load_tsv(test_path)
    print(
        f"Labeled rows: {len(labeled_df)}, "
        f"unlabeled rows: {len(unlabeled_df)}, "
        f"test rows: {len(test_df)}"
    )

    labels = labeled_df["sentiment"].astype(int).to_numpy()
    run_timestamp = datetime.now().isoformat(timespec="microseconds")

    classifier_label = "RandomForest" if args.classifier == "random_forest" else "LogisticRegression"
    if args.classifier == "linear_svc":
        classifier_label = "LinearSVC"

    if args.feature == "tfidf":
        print("\n=== Using TF-IDF with N-gram features ===")
        print("Preprocessing text (keeping negation words)...")
        
        train_texts = preprocess_for_tfidf(labeled_df["review"])
        test_texts = preprocess_for_tfidf(test_df["review"])
        
        print(f"Building TF-IDF features with n-gram ({args.ngram_min}-{args.ngram_max})...")
        train_features, test_features, vectorizer = build_tfidf_features(
            train_texts, test_texts, args
        )
        print(f"TF-IDF feature shape: {train_features.shape}")
        
        feature_name = f"tfidf_ngram{args.ngram_min}-{args.ngram_max}_max{args.max_features}"
        print(f"\nEvaluating {feature_name} + {classifier_label}...")
        scores, mean_auc = evaluate_with_cv(
            train_features,
            labels,
            args.folds,
            args.classifier,
            args.lr_c,
            args.lr_max_iter,
            args.rf_trees,
        )
        print(f"TF-IDF ROC-AUC: {mean_auc:.5f} (std={np.std(scores):.5f})")
        
        log_attempt(
            timestamp=run_timestamp,
            feature_type=feature_name,
            scores=scores,
            mean_auc=mean_auc,
            args=args,
            labeled_rows=len(labeled_df),
            unlabeled_rows=len(unlabeled_df),
            submission_path=None,
        )
        
        print("\nTraining final model on all data...")
        submission_scores = fit_and_predict_submission(
            train_features,
            labels,
            test_features,
            args.classifier,
            args.lr_c,
            args.lr_max_iter,
            args.rf_trees,
        )
        
    else:
        print("\n=== Using Word2Vec features ===")
        print("Tokenizing reviews for Word2Vec training...")
        w2v_sentences = preprocess_reviews(
            pd.concat(
                [labeled_df["review"], unlabeled_df["review"], test_df["review"]],
                ignore_index=True,
            ),
            remove_stopwords=False,
        )

        print("Loading or training Word2Vec...")
        model, cache_path, trained_now = load_or_train_word2vec(w2v_sentences, args)
        action = "trained and saved" if trained_now else "loaded from cache"
        print(f"Word2Vec {action}: {cache_path}")
        print(f"Word2Vec vocabulary size: {len(model.wv)}")

        print("Tokenizing labeled and test reviews for feature construction...")
        labeled_tokens = preprocess_reviews(labeled_df["review"], remove_stopwords=True)
        test_tokens = preprocess_reviews(test_df["review"], remove_stopwords=True)

        print(f"\nEvaluating mean Word2Vec + {classifier_label}...")
        mean_features = build_average_vectors(labeled_tokens, model)
        mean_scores, mean_auc = evaluate_with_cv(
            mean_features,
            labels,
            args.folds,
            args.classifier,
            args.lr_c,
            args.lr_max_iter,
            args.rf_trees,
        )
        print(f"Mean embeddings ROC-AUC: {mean_auc:.5f} (std={np.std(mean_scores):.5f})")
        
        log_attempt(
            timestamp=run_timestamp,
            feature_type="mean_embeddings",
            scores=mean_scores,
            mean_auc=mean_auc,
            args=args,
            labeled_rows=len(labeled_df),
            unlabeled_rows=len(unlabeled_df),
            w2v_model_path=cache_path,
            submission_path=None,
        )

        print(f"\nClustering word vectors with KMeans (k={args.kmeans_clusters})...")
        _, vocab_words, cluster_labels = fit_kmeans_on_word_vectors(model, args.kmeans_clusters)
        word_to_cluster = dict(zip(vocab_words, cluster_labels))

        print(f"Evaluating bag-of-centroids + {classifier_label}...")
        kmeans_train_features = build_bag_of_centroids(labeled_tokens, word_to_cluster, args.kmeans_clusters)
        kmeans_scores, kmeans_auc = evaluate_with_cv(
            kmeans_train_features,
            labels,
            args.folds,
            args.classifier,
            args.lr_c,
            args.lr_max_iter,
            args.rf_trees,
        )
        print(f"KMeans centroid features ROC-AUC: {kmeans_auc:.5f} (std={np.std(kmeans_scores):.5f})")
        
        log_attempt(
            timestamp=run_timestamp,
            feature_type=f"kmeans_centroids_k{args.kmeans_clusters}",
            scores=kmeans_scores,
            mean_auc=kmeans_auc,
            args=args,
            labeled_rows=len(labeled_df),
            unlabeled_rows=len(unlabeled_df),
            w2v_model_path=cache_path,
            submission_path=None,
        )

        if mean_auc >= kmeans_auc:
            selected_name = "mean_embeddings"
            train_features = mean_features
            test_features = build_average_vectors(test_tokens, model)
            selected_auc = mean_auc
        else:
            selected_name = f"kmeans_centroids_k{args.kmeans_clusters}"
            train_features = kmeans_train_features
            test_features = build_bag_of_centroids(test_tokens, word_to_cluster, args.kmeans_clusters)
            selected_auc = kmeans_auc

        print(f"\nSelected feature set: {selected_name} (CV ROC-AUC = {selected_auc:.5f})")
        
        print("\nTraining final model on all data...")
        submission_scores = fit_and_predict_submission(
            train_features,
            labels,
            test_features,
            args.classifier,
            args.lr_c,
            args.lr_max_iter,
            args.rf_trees,
        )

    if args.classifier == "logistic_regression":
        submission_tag = f"{args.classifier}_c{args.lr_c}_{feature_name if args.feature == 'tfidf' else selected_name}"
    elif args.classifier == "linear_svc":
        submission_tag = f"{args.classifier}_c{args.lr_c}_{feature_name if args.feature == 'tfidf' else selected_name}"
    else:
        submission_tag = f"{args.classifier}_trees{args.rf_trees}_{feature_name if args.feature == 'tfidf' else selected_name}"
    
    submission_path = build_submission_path(run_timestamp, submission_tag)
    submission_df = pd.DataFrame({"id": test_df["id"], "sentiment": submission_scores})
    submission_df.to_csv(submission_path, index=False, quoting=2)
    print(f"Saved submission to: {submission_path}")
    
    log_attempt(
        timestamp=run_timestamp,
        feature_type=f"{feature_name if args.feature == 'tfidf' else selected_name}_submission",
        scores=[],
        mean_auc=mean_auc if args.feature == 'tfidf' else selected_auc,
        args=args,
        labeled_rows=len(labeled_df),
        unlabeled_rows=len(unlabeled_df),
        submission_path=submission_path,
    )
    print(f"Updated attempt log at: {ATTEMPT_LOG_PATH}")


if __name__ == "__main__":
    main()
