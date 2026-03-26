from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path
from gensim.models import Word2Vec
import numpy as np

# Create a Word2Vec Vectorizer for sklearn.pipline
class w2v_Vectorizer:
	def __init__(self, vector_size=100, window=5, min_count=1, workers=1, epochs=20):
		self.model = None
		self.vector_size = vector_size
		self.window = window
		self.min_count = min_count
		self.workers = workers
		self.epochs = epochs

	def fit(self, X, y=None):
		sentences = []
		for text in X:
			sentences.append(text.split())
		self.model = Word2Vec(
			sentences=sentences,
			vector_size=self.vector_size,
			window=self.window,
			min_count=self.min_count,
			workers=self.workers,
			epochs=self.epochs
		)
		return self
	
	def transform(self, X):
		vectors = []
		for text in X:
			words = text.split()
			words_vectors = []
			for w in words:
				if w in self.model.wv:
					v = self.model.wv[w]
					words_vectors.append(v)
			if len(words_vectors)> 0:
				vectors.append(np.mean(words_vectors, axis=0))
			else:
				vectors.append(np.zeros(self.vector_size))
		return np.array(vectors)
	
# Create a scikit-learn pipeline with Word2Vec Vectorier and LogisticRegression.
def create_w2v_pipeline(vector_size=100, window=5, min_count=1, workers=1, epochs=20):
	pipeline = Pipeline([
        ('w2v', w2v_Vectorizer(vector_size, window, min_count, workers, epochs)),
        ('clf', LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"))
    ])
	return pipeline

# Train the model on the training DataFrame.
def train_w2v_model(train_df):
    pipeline = create_w2v_pipeline()
    X_train = train_df['text'].fillna("")
    y_train = train_df['label']
    print("Training Word2Vec model...")
    pipeline.fit(X_train, y_train)
    return pipeline

# Return predictions for the test DataFrame.
def get_w2v_predictions(model, test_df):
    X_test = test_df['text'].fillna("")
    return model.predict(X_test)

# Calculate metrics and generate a confusion matrix plot.
def evaluate_w2v_model(y_true, y_pred, output_path="outputs/w2v_confusion_matrix.png"):
    # Print metrics
    print("\n--- Word2Vec Model Evaluation ---")
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