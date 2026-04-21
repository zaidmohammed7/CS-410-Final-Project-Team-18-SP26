from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    precision_recall_curve,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
from pathlib import Path

def create_tfidf_pipeline(model_type='logistic'):
    """
    Create a scikit-learn pipeline with TfidfVectorizer and a classifier.
    Supports 'logistic' or 'svm'.
    Enables bigrams by default (ngram_range=(1,2)).
    """
    if model_type == 'logistic':
        clf = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'svm':
        clf = LinearSVC(random_state=42, max_iter=2000, dual='auto')
    else:
        raise ValueError("Unsupported model_type. Use 'logistic' or 'svm'.")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
        ('clf', clf)
    ])
    return pipeline

def train_tfidf_model(train_df, model_type='logistic'):
    """
    Train the model on the training DataFrame.
    """
    pipeline = create_tfidf_pipeline(model_type=model_type)
    X_train = train_df['text'].fillna("")
    y_train = train_df['label']
    
    print(f"Training TF-IDF model ({model_type})...")
    pipeline.fit(X_train, y_train)
    return pipeline

def get_tfidf_predictions(model, test_df):
    """
    Return predictions for the test DataFrame.
    """
    X_test = test_df['text'].fillna("")
    return model.predict(X_test)

def evaluate_tfidf_model(y_true, y_pred, model, test_df, model_name="tfidf", output_dir="outputs"):
    """
    Calculate metrics and generate plots (Confusion Matrix and PR Curve).
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Print metrics
    print(f"\n--- {model_name.upper()} Model Evaluation ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Ham", "Spam"]))
    
    # Visual Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / f"{model_name}_confusion_matrix.png")
    plt.close()

    # Precision-Recall Curve
    # SVM and Logistic Regression handle scores differently
    X_test = test_df['text'].fillna("")
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
