from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path

def create_pipeline():
    """
    Create a scikit-learn pipeline with TfidfVectorizer and LogisticRegression.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf', LogisticRegression(random_state=42))
    ])
    return pipeline

def train_model(train_df):
    """
    Train the model on the training DataFrame.
    """
    pipeline = create_pipeline()
    X_train = train_df['text'].fillna("")
    y_train = train_df['label']
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    return pipeline

def get_predictions(model, test_df):
    """
    Return predictions for the test DataFrame.
    """
    X_test = test_df['text'].fillna("")
    return model.predict(X_test)

def evaluate_model(y_true, y_pred, output_path="outputs/confusion_matrix.png"):
    """
    Calculate metrics and generate a confusion matrix plot.
    """
    # Print metrics
    print("\n--- Model Evaluation ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Ham", "Spam"]))
    
    # Visual Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
    
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to '{output_path}'.")
