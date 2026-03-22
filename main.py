from pathlib import Path

from src.preprocessor import load_enron_dataset, load_spamassassin_dataset, preprocess

TRAIN_CSV = Path("dataset/train/enron_spam_data.csv")
TEST_DIR = Path("dataset/test")

def main():
    print("Loading datasets...")
    train_df = load_enron_dataset(TRAIN_CSV)
    test_df = load_spamassassin_dataset(TEST_DIR)

    print(f"  Train samples: {len(train_df)} | Spam: {train_df['label'].sum()}")
    print(f"  Test samples:  {len(test_df)}  | Spam: {test_df['label'].sum()}")

    print("Preprocessing...")
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    # Check preprocessing delete later
    print("\n--- Train sample (ham) ---")
    print(train_df[train_df["label"] == 0]["text"].iloc[0][:300])

    print("\n--- Train sample (spam) ---")
    print(train_df[train_df["label"] == 1]["text"].iloc[0][:300])

    print("\n--- Test sample (ham) ---")
    print(test_df[test_df["label"] == 0]["text"].iloc[0][:300])

    print("\n--- Test sample (spam) ---")
    print(test_df[test_df["label"] == 1]["text"].iloc[0][:300])


if __name__ == "__main__":
    main()
