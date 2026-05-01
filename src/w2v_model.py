from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
from pathlib import Path
try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
import numpy as np

from src.TF_IDF_model import STRUCTURAL_COLS, get_feature_columns


# Custom sklearn-compatible Word2Vec Vectorizer
class w2v_Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=50, window=5, min_count=5, workers=4, epochs=5):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model = None

    def fit(self, X, y=None):
        sentences = [str(text).split() for text in X]
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            seed=12345
        )
        return self

    def transform(self, X):
        vectors = []
        for text in X:
            words = str(text).split()
            word_vecs = [self.model.wv[w] for w in words if w in self.model.wv]
            if word_vecs:
                vectors.append(np.mean(word_vecs, axis=0))
            else:
                vectors.append(np.zeros(self.vector_size))
        return np.array(vectors)


def create_w2v_pipeline(vector_size=50, window=5, min_count=5,
                         workers=4, epochs=5, use_structural=True):
    """Create a pipeline with Word2Vec embeddings, optional structural features,
    and a balanced Logistic Regression classifier."""
    transformers = [
        ('w2v', w2v_Vectorizer(vector_size, window, min_count, workers, epochs), 'text'),
    ]
    if use_structural:
        transformers.append(('structural', StandardScaler(), STRUCTURAL_COLS))

    pipeline = Pipeline([
        ('features', ColumnTransformer(transformers=transformers)),
        ('clf', LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"))
    ])
    return pipeline


def train_w2v_model(train_df, use_structural=True):
    """Train the Word2Vec pipeline on the training DataFrame."""
    pipeline = create_w2v_pipeline(use_structural=use_structural)
    cols = get_feature_columns(use_structural)
    X_train = train_df[cols]
    y_train = train_df['label']
    print("Training Word2Vec model...")
    pipeline.fit(X_train, y_train)
    return pipeline


def get_w2v_predictions(model, test_df, use_structural=True):
    """Return predictions for the test DataFrame."""
    cols = get_feature_columns(use_structural)
    return model.predict(test_df[cols])


def evaluate_w2v_model(y_true, y_pred, model, test_df,
                        model_name="w2v", output_dir="outputs",
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
    y_score = model.predict_proba(X_test)[:, 1]

    plt.figure(figsize=(8, 6))
    display = PrecisionRecallDisplay.from_predictions(y_true, y_score, name=model_name)
    display.ax_.set_title(f"Precision-Recall Curve - {model_name}")
    plt.savefig(output_path / f"{model_name}_pr_curve.png")
    plt.close()

    print(f"Plots saved to '{output_path}'.")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}