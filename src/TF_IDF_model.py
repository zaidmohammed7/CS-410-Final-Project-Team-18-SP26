from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
from pathlib import Path

STRUCTURAL_COLS = [
    "url_count", "url_density", "exclamation_count", "question_count",
    "uppercase_ratio", "digit_ratio", "punctuation_ratio", "all_caps_word_ratio",
    "html_marker_count", "subject_char_count", "body_char_count",
    "sender_char_count", "subject_word_count", "body_word_count",
    "text_word_count", "sender_word_count", "sender_has_email",
    "sender_has_domain", "sender_digit_ratio", "sender_uppercase_ratio",
    "sender_punctuation_ratio",
]


def get_feature_columns(use_structural=True):
    """Return the list of DataFrame columns needed by the pipeline."""
    cols = ['text']
    if use_structural:
        cols += STRUCTURAL_COLS
    return cols


def create_tfidf_pipeline(model_type='logistic', ngram_range=(1, 2),
                           use_structural=True, vectorizer_type='tfidf'):
    """
    Create a pipeline with text vectorization, optional structural features,
    and a classifier.

    Parameters:
        model_type: 'logistic' or 'svm'
        ngram_range: e.g. (1,1) for unigrams, (1,2) for uni+bigrams
        use_structural: whether to append the 21 structural metadata features
        vectorizer_type: 'tfidf' or 'count' (raw keyword counts)
    """
    if model_type == 'logistic':
        clf = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'svm':
        clf = LinearSVC(random_state=42, max_iter=2000, dual='auto')
    else:
        raise ValueError("Unsupported model_type. Use 'logistic' or 'svm'.")

    if vectorizer_type == 'tfidf':
        text_vec = TfidfVectorizer(max_features=5000, stop_words='english',
                                   ngram_range=ngram_range)
    elif vectorizer_type == 'count':
        text_vec = CountVectorizer(max_features=5000, stop_words='english',
                                   ngram_range=ngram_range)
    else:
        raise ValueError("Unsupported vectorizer_type.")

    transformers = [('text', text_vec, 'text')]
    if use_structural:
        transformers.append(('structural', StandardScaler(), STRUCTURAL_COLS))

    pipeline = Pipeline([
        ('features', ColumnTransformer(transformers=transformers)),
        ('clf', clf)
    ])
    return pipeline


def train_tfidf_model(train_df, model_type='logistic', ngram_range=(1, 2),
                       use_structural=True, vectorizer_type='tfidf'):
    """Train the model on the training DataFrame."""
    pipeline = create_tfidf_pipeline(
        model_type=model_type, ngram_range=ngram_range,
        use_structural=use_structural, vectorizer_type=vectorizer_type
    )
    cols = get_feature_columns(use_structural)
    X_train = train_df[cols]
    y_train = train_df['label']

    desc = f"{vectorizer_type.upper()} {model_type} ngram={ngram_range}"
    if use_structural:
        desc += " +structural"
    print(f"Training: {desc} ...")
    pipeline.fit(X_train, y_train)
    return pipeline


def get_tfidf_predictions(model, test_df, use_structural=True):
    """Return predictions for the test DataFrame."""
    cols = get_feature_columns(use_structural)
    return model.predict(test_df[cols])


def evaluate_tfidf_model(y_true, y_pred, model, test_df,
                          model_name="tfidf", output_dir="outputs",
                          use_structural=True):
    """Calculate metrics and generate plots (Confusion Matrix and PR Curve)."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n--- {model_name.upper()} Model Evaluation ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Ham", "Spam"],
                                zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / f"{model_name}_confusion_matrix.png")
    plt.close()

    # Precision-Recall Curve
    cols = get_feature_columns(use_structural)
    X_test = test_df[cols]
    if hasattr(model['clf'], "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = model.predict_proba(X_test)[:, 1]

    plt.figure(figsize=(8, 6))
    display = PrecisionRecallDisplay.from_predictions(y_true, y_score, name=model_name)
    display.ax_.set_title(f"Precision-Recall Curve - {model_name}")
    plt.savefig(output_path / f"{model_name}_pr_curve.png")
    plt.close()

    print(f"Plots saved to '{output_path}'.")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
