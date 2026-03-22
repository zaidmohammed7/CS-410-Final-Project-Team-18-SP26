import re
import email
import email.message
import pandas as pd
from pathlib import Path

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

# Load the Enron CSV training dataset
def load_enron_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df = df.rename(columns={"Subject": "subject", "Message": "body", "Spam/Ham": "label"})
    df["sender"] = "" # Enron doesn't have sender data so just adding empty
    df["label"] = df["label"].str.lower().map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
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

# Apply text cleaning to subject and and body and combine into one feature column
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["subject"] = df["subject"].fillna("").apply(clean_text)
    df["body"] = df["body"].fillna("").apply(clean_text)
    df["text"] = df["subject"] + " " + df["body"]
    return df
