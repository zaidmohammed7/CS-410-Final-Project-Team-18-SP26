from pathlib import Path
import pandas as pd

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

def main():
    ensure_archive_extracted(TRAIN_CSV, TRAIN_ARCHIVE, DATASET_DIR)
    ensure_archive_extracted(TEST_DIR, TEST_ARCHIVE, DATASET_DIR)

    print("Loading datasets...")
    train_df = load_training_dataset(TRAIN_CSV)
    test_df = load_spamassassin_dataset(TEST_DIR)

    print(f"  Train samples: {len(train_df)} | Spam: {train_df['label'].sum()}")
    print(f"  Test samples:  {len(test_df)}  | Spam: {test_df['label'].sum()}")

    print("Preprocessing...")
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)
    train_df, test_df = prepare_datasets(train_df, test_df)

    # Combine datasets for EDA purposes as requested ("across datasets")
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    run_eda(combined_df)

    # Model Pipeline
    print("\nTraining set prepared for model...")
    #tfidf_model = train_tfidf_model(train_df)
    w2v_model = train_w2v_model(train_df)
    
    # Prediction on test set
    print("Generating predictions on test set...")
    #tfidf_predictions = get_tfidf_predictions(tfidf_model, test_df)
    w2v_predictions = get_w2v_predictions(w2v_model, test_df)
    
    # Evaluation
    #evaluate_tfidf_model(test_df['label'], tfidf_predictions)
    evaluate_w2v_model(test_df['label'], w2v_predictions)

    print("\n--- Pipeline Complete ---")
    print("Outputs generated in 'outputs/' directory.")

if __name__ == "__main__":
    main()
