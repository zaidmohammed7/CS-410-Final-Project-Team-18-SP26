from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

from src.preprocessor import (
    ensure_archive_extracted,
    load_training_dataset,
    load_spamassassin_dataset,
    preprocess,
    prepare_datasets,
)
from src.eda import run_eda
from src.TF_IDF_model import (
    train_tfidf_model, get_tfidf_predictions, evaluate_tfidf_model,
    get_feature_columns,
)
from src.w2v_model import train_w2v_model, get_w2v_predictions, evaluate_w2v_model

DATASET_DIR = Path("dataset")
TRAIN_CSV = DATASET_DIR / "train" / "processed_data.csv"
TEST_DIR = DATASET_DIR / "test"
TRAIN_ARCHIVE = DATASET_DIR / "train.zip"
TEST_ARCHIVE = DATASET_DIR / "test.zip"


def run_cross_validation(model, X, y, cv=5):
    """Perform K-fold cross-validation and print results."""
    print(f"\n  {cv}-fold Cross-Validation...")
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    print(f"  F1-Score Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    return scores.mean(), scores.std()


def run_pipeline(name, train_df, test_df, *, model_type='logistic',
                 ngram_range=(1, 2), use_structural=True,
                 vectorizer_type='tfidf', do_cv=True):
    """Train, cross-validate, predict, and evaluate a single TF-IDF-family model."""
    print(f"\n{'='*60}")
    print(f"  Pipeline: {name}")
    print(f"{'='*60}")

    model = train_tfidf_model(
        train_df, model_type=model_type, ngram_range=ngram_range,
        use_structural=use_structural, vectorizer_type=vectorizer_type
    )

    if do_cv:
        cols = get_feature_columns(use_structural)
        run_cross_validation(model, train_df[cols], train_df['label'])

    preds = get_tfidf_predictions(model, test_df, use_structural=use_structural)
    # Use the pipeline name (lowercased, no spaces) for file naming
    file_tag = name.lower().replace(" ", "_").replace("+", "").replace("(", "").replace(")", "")
    metrics = evaluate_tfidf_model(
        test_df['label'], preds, model, test_df,
        model_name=file_tag, use_structural=use_structural
    )
    return metrics


def main():
    ensure_archive_extracted(TRAIN_CSV, TRAIN_ARCHIVE, DATASET_DIR)
    ensure_archive_extracted(TEST_DIR, TEST_ARCHIVE, DATASET_DIR)

    print("Loading datasets...")
    train_df = load_training_dataset(TRAIN_CSV)
    test_df = load_spamassassin_dataset(TEST_DIR)

    print(f"  Train samples: {len(train_df)} | Spam: {train_df['label'].sum()}")
    print(f"  Test samples:  {len(test_df)}  | Spam: {test_df['label'].sum()}")

    print("Preprocessing (including Lemmatization)...")
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)
    train_df, test_df = prepare_datasets(train_df, test_df)

    # EDA on combined corpus
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    run_eda(combined_df)

    # ------------------------------------------------------------------
    # 1. Keyword Baseline  (CountVectorizer, unigrams only, NO structural)
    # ------------------------------------------------------------------
    run_pipeline(
        "Keyword Baseline", train_df, test_df,
        model_type='logistic', ngram_range=(1, 1),
        use_structural=False, vectorizer_type='count', do_cv=True
    )

    # ------------------------------------------------------------------
    # 2. TF-IDF Unigram Only  (no structural) — for ngram comparison
    # ------------------------------------------------------------------
    run_pipeline(
        "TFIDF Unigram", train_df, test_df,
        model_type='logistic', ngram_range=(1, 1),
        use_structural=False, vectorizer_type='tfidf', do_cv=True
    )

    # ------------------------------------------------------------------
    # 3. Logistic Regression  (TF-IDF bigrams + structural features)
    # ------------------------------------------------------------------
    run_pipeline(
        "Logistic", train_df, test_df,
        model_type='logistic', ngram_range=(1, 2),
        use_structural=True, vectorizer_type='tfidf', do_cv=True
    )

    # ------------------------------------------------------------------
    # 4. Linear SVM  (TF-IDF bigrams + structural features)
    # ------------------------------------------------------------------
    run_pipeline(
        "SVM", train_df, test_df,
        model_type='svm', ngram_range=(1, 2),
        use_structural=True, vectorizer_type='tfidf', do_cv=True
    )

    # ------------------------------------------------------------------
    # 5. Word2Vec Neural Embeddings  (+ structural features)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Pipeline: Word2Vec")
    print(f"{'='*60}")
    try:
        w2v_model = train_w2v_model(train_df, use_structural=True)
        w2v_preds = get_w2v_predictions(w2v_model, test_df, use_structural=True)
        evaluate_w2v_model(test_df['label'], w2v_preds, w2v_model, test_df,
                           use_structural=True)
    except Exception as e:
        print(f"Skipping Word2Vec due to error: {e}")
        import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print("  All pipelines complete.")
    print("  Outputs saved in 'outputs/' directory.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
