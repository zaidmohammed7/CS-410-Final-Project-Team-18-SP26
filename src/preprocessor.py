import re
import email
import email.message
import zipfile
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
PUNCTUATION_PATTERN = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]")
EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")


def ensure_archive_extracted(target_path: Path, archive_path: Path, dataset_root: Path) -> None:
    if target_path.exists() or not archive_path.exists():
        return

    with zipfile.ZipFile(archive_path, "r") as archive:
        members = [
            member
            for member in archive.namelist()
            if member and not member.startswith("__MACOSX/")
        ]
        target_dir_name = target_path.parent.name if target_path.suffix else target_path.name
        has_nested_target_dir = any(member.startswith(f"{target_dir_name}/") for member in members)
        extract_dir = dataset_root if has_nested_target_dir else target_path.parent
        extract_dir.mkdir(parents=True, exist_ok=True)
        archive.extractall(extract_dir, members=members)

    if not target_path.exists():
        raise FileNotFoundError(
            f"Extracted '{archive_path}' but could not find expected path '{target_path}'."
        )

# Parse raw SpamAssassin email file into its components
def parse_raw_email(file_path: Path) -> dict:
    with open(file_path, "rb") as f:
        msg = email.message_from_binary_file(f)

    subject = msg.get("Subject", "") or ""
    sender = msg.get("From", "") or ""
    body = _extract_body(msg)

    return {"subject": subject, "sender": sender, "body": body}

def _decode_payload(part: email.message.Message) -> str:
    payload = part.get_payload(decode=True)
    if not payload:
        return ""
    return payload.decode("utf-8", errors="replace")


# Recursively extract text from an email, preferring plain text over HTML
def _extract_body(msg: email.message.Message) -> str:
    plain_parts = []
    html_parts = []

    parts = list(msg.walk()) if msg.is_multipart() else [msg]
    for part in parts:
        content_type = part.get_content_type()
        if content_type == "text/plain":
            plain_parts.append(_decode_payload(part))
        elif content_type == "text/html":
            html_parts.append(_decode_payload(part))

    # Use plain text if available otherwise fall back to HTML with tags stripped later by clean_text
    return " ".join(plain_parts) if plain_parts else " ".join(html_parts)

# Load a CSV dataset for training, supporting both the original Enron schema
# and the processed TREC schema now stored in dataset/train.
def load_training_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    unnamed_columns = [column for column in df.columns if str(column).startswith("Unnamed:")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    if {"Subject", "Message", "Spam/Ham"}.issubset(df.columns):
        df = df.rename(columns={"Subject": "subject", "Message": "body", "Spam/Ham": "label"})
        df["sender"] = ""
        df["label"] = df["label"].str.lower().map({"ham": 0, "spam": 1})
    elif {"label", "subject", "email_from", "message"}.issubset(df.columns):
        df = df.rename(columns={"email_from": "sender", "message": "body"})
    else:
        raise ValueError(
            f"Unsupported training dataset schema in {csv_path}. "
            "Expected either Enron CSV columns or processed TREC columns."
        )

    df["subject"] = df["subject"].fillna("")
    df["sender"] = df["sender"].fillna("")
    df["body"] = df["body"].fillna("")
    return df[["subject", "sender", "body", "label"]]

# Load SpamAssassin test dataset from raw email files
def load_spamassassin_dataset(data_dir: Path) -> pd.DataFrame:
    records = []

    for folder, label in [("easy_ham", 0), ("spam_2", 1)]:
        folder_path = data_dir / folder
        if not folder_path.exists():
            continue
        for file_path in sorted(folder_path.iterdir()):
            if file_path.is_file() and not file_path.name.startswith("."):
                parsed = parse_raw_email(file_path)
                parsed["label"] = label
                records.append(parsed)

    return pd.DataFrame(records, columns=["subject", "sender", "body", "label"])

# Normalize string by lowercasing and stripping HTML tags and extra whitespace
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text) # remove HTML tags
    text = re.sub(r"http\S+|www\.\S+", " URL ", text) # replace URLs
    text = re.sub(r"[^a-zA-Z0-9\s\!\?\.\,]", " ", text) # keep basic punctuation
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def _count_all_caps_words(text: str) -> int:
    return sum(1 for token in text.split() if len(token) > 1 and token.isupper())


def extract_structural_features(subject: str, body: str, sender: str = "") -> dict:
    subject = subject if isinstance(subject, str) else ""
    body = body if isinstance(body, str) else ""
    sender = sender if isinstance(sender, str) else ""
    combined = f"{subject} {body}".strip()

    letters = [char for char in combined if char.isalpha()]
    uppercase_letters = sum(1 for char in letters if char.isupper())
    digits = sum(1 for char in combined if char.isdigit())
    punctuation_count = len(PUNCTUATION_PATTERN.findall(combined))
    exclamation_count = combined.count("!")
    question_count = combined.count("?")
    url_count = len(URL_PATTERN.findall(combined))
    html_marker_count = body.lower().count("content type text html") + body.lower().count("text/html")
    sender_letters = [char for char in sender if char.isalpha()]
    sender_digits = sum(1 for char in sender if char.isdigit())
    sender_punctuation_count = len(PUNCTUATION_PATTERN.findall(sender))
    sender_has_email = 1.0 if EMAIL_PATTERN.search(sender) else 0.0
    sender_has_domain = 1.0 if "@" in sender else 0.0
    all_caps_word_count = _count_all_caps_words(combined)

    subject_tokens = subject.split()
    body_tokens = body.split()
    combined_tokens = combined.split()
    sender_tokens = sender.split()

    return {
        "url_count": float(url_count),
        "url_density": _safe_ratio(url_count, len(combined_tokens)),
        "exclamation_count": float(exclamation_count),
        "question_count": float(question_count),
        "uppercase_ratio": _safe_ratio(uppercase_letters, len(letters)),
        "digit_ratio": _safe_ratio(digits, len(combined)),
        "punctuation_ratio": _safe_ratio(punctuation_count, len(combined)),
        "all_caps_word_ratio": _safe_ratio(all_caps_word_count, len(combined_tokens)),
        "html_marker_count": float(html_marker_count),
        "subject_char_count": float(len(subject)),
        "body_char_count": float(len(body)),
        "sender_char_count": float(len(sender)),
        "subject_word_count": float(len(subject_tokens)),
        "body_word_count": float(len(body_tokens)),
        "text_word_count": float(len(combined_tokens)),
        "sender_word_count": float(len(sender_tokens)),
        "sender_has_email": sender_has_email,
        "sender_has_domain": sender_has_domain,
        "sender_digit_ratio": _safe_ratio(sender_digits, len(sender)),
        "sender_uppercase_ratio": _safe_ratio(
            sum(1 for char in sender_letters if char.isupper()),
            len(sender_letters),
        ),
        "sender_punctuation_ratio": _safe_ratio(sender_punctuation_count, len(sender)),
    }

# Apply text cleaning to subject and and body and combine into one feature column
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["raw_subject"] = df["subject"].fillna("")
    df["raw_body"] = df["body"].fillna("")
    df["raw_sender"] = df["sender"].fillna("")

    structural_features = df.apply(
        lambda row: extract_structural_features(row["raw_subject"], row["raw_body"], row["raw_sender"]),
        axis=1,
        result_type="expand",
    )

    df["subject"] = df["raw_subject"].apply(clean_text)
    df["body"] = df["raw_body"].apply(clean_text)
    df["sender"] = df["raw_sender"].apply(clean_text)
    df["text"] = df["subject"] + " " + df["body"]
    df["text"] = df["text"].apply(lambda text: WHITESPACE_PATTERN.sub(" ", text).strip())
    df = pd.concat([df, structural_features], axis=1)
    return df


def clean_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    *,
    drop_empty: bool = True,
    deduplicate: bool = True,
    dedupe_subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = df.copy()
    initial_rows = len(df)

    if "text" not in df.columns:
        raise ValueError("clean_dataset expects a preprocessed DataFrame with a 'text' column.")

    if drop_empty:
        non_empty_mask = df["text"].fillna("").str.strip().str.len() > 0
        dropped_empty = int((~non_empty_mask).sum())
        df = df.loc[non_empty_mask].copy()
    else:
        dropped_empty = 0

    removed_duplicates = 0
    conflicting_groups = 0
    if deduplicate:
        conflicting_groups = int((df.groupby("text")["label"].nunique() > 1).sum())
        subset = dedupe_subset if dedupe_subset is not None else ["text", "label"]
        duplicate_mask = df.duplicated(subset=subset, keep="first")
        removed_duplicates = int(duplicate_mask.sum())
        df = df.loc[~duplicate_mask].copy().reset_index(drop=True)

    print(
        f"{dataset_name}: removed {dropped_empty} empty, "
        f"{removed_duplicates} exact duplicate rows. "
        f"Conflicting text groups still present: {conflicting_groups}. "
        f"Remaining: {len(df)} of {initial_rows}."
    )
    return df


def summarize_label_diversity(df: pd.DataFrame, dataset_name: str) -> None:
    if "label" not in df.columns or "text" not in df.columns:
        return

    label_summary = (
        df.groupby("label")["text"]
        .agg(total_rows="size", unique_texts="nunique")
        .reset_index()
    )
    label_summary["unique_ratio"] = label_summary["unique_texts"] / label_summary["total_rows"]

    print(f"{dataset_name} label diversity:")
    for _, row in label_summary.iterrows():
        print(
            f"  label={int(row['label'])}: rows={int(row['total_rows'])}, "
            f"unique_texts={int(row['unique_texts'])}, "
            f"unique_ratio={row['unique_ratio']:.4f}"
        )


def prepare_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = clean_dataset(
        train_df,
        "Train",
        drop_empty=True,
        deduplicate=True,
        dedupe_subset=["text", "label"],
    )
    test_df = clean_dataset(
        test_df,
        "Test",
        drop_empty=True,
        deduplicate=True,
        dedupe_subset=["text", "label"],
    )

    summarize_label_diversity(train_df, "Train")
    summarize_label_diversity(test_df, "Test")
    return train_df, test_df
