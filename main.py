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
from src.TF_IDF_model import train_tfidf_model, get_tfidf_predictions, evaluate_tfidf_model
from src.w2v_model import train_w2v_model, get_w2v_predictions, evaluate_w2v_model

DATASET_DIR = Path("dataset")
TRAIN_CSV = DATASET_DIR / "train" / "processed_data.csv"
TEST_DIR = DATASET_DIR / "test"
TRAIN_ARCHIVE = DATASET_DIR / "train.zip"
TEST_ARCHIVE = DATASET_DIR / "test.zip"

def run_cross_validation(model, X, y, cv=5):
    """
    Perform K-fold cross-validation and print results.
    """
    print(f"\nPerforming {cv}-fold Cross-Validation...")
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    print(f"  F1-Score Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

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

    # Combine datasets for EDA purposes
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    run_eda(combined_df)

    # 1. Logistic Regression Pipeline
    print("\n--- Pipeline 1: Logistic Regression ---")
    tfidf_logistic = train_tfidf_model(train_df, model_type='logistic')
    
    # Cross-validation on training set
    run_cross_validation(tfidf_logistic, train_df['text'].fillna(""), train_df['label'])
    
    # Prediction and Evaluation
    tfidf_log_preds = get_tfidf_predictions(tfidf_logistic, test_df)
    evaluate_tfidf_model(test_df['label'], tfidf_log_preds, tfidf_logistic, test_df, model_name="logistic")

    # 2. Linear SVM Pipeline
    print("\n--- Pipeline 2: Linear SVM ---")
    tfidf_svm = train_tfidf_model(train_df, model_type='svm')
    
    # Cross-validation on training set
    run_cross_validation(tfidf_svm, train_df['text'].fillna(""), train_df['label'])
    
    # Prediction and Evaluation
    tfidf_svm_preds = get_tfidf_predictions(tfidf_svm, test_df)
    evaluate_tfidf_model(test_df['label'], tfidf_svm_preds, tfidf_svm, test_df, model_name="svm")

    # 3. Word2Vec Pipeline
    print("\n--- Pipeline 3: Word2Vec ---")
    try:
        w2v_model = train_w2v_model(train_df)
        
        # Cross-validation on training set (Slower, commented out for fast generation)
        # run_cross_validation(w2v_model, train_df['text'].fillna(""), train_df['label'])
        
        # Prediction and Evaluation
        w2v_preds = get_w2v_predictions(w2v_model, test_df)
        evaluate_w2v_model(test_df['label'], w2v_preds, w2v_model, test_df)
    except Exception as e:
        print(f"Skipping Word2Vec due to error: {e}")

    print("\n--- Pipeline Complete ---")
    print("Outputs (Confusion Matrices, PR Curves, EDA) generated in 'outputs/' directory.")

if __name__ == "__main__":
    main()
