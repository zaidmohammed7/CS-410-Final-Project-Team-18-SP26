# CS 410 Final Project: Multi-Faceted Spam Email Detection

This repository contains the final implementation of Team 18's Text-Based Spam Email Detection System. To address the challenge of cross-corpus domain shift (feature sparsity), this system employs a comparative architecture that evaluates emails through three distinct approaches:
1. **Logistic Regression** (TF-IDF + Bigrams + Structural Metadata)
2. **Linear SVM** (TF-IDF + Bigrams + Structural Metadata)
3. **Word2Vec Neural Embeddings** (Latent Space Centroids)

## 📁 Dataset Setup

- Training archive: `dataset/train.zip` (Processed TREC 2007)
- Testing archive: `dataset/test.zip` (Apache SpamAssassin)

*Note: You do not need to extract these manually. The pipeline script (`main.py`) will automatically verify and extract the archives before beginning data ingestion if the expected paths do not exist.*

## ⚙️ Environment Configuration

Because the Word2Vec model relies on the `gensim` library, **Python 3.12 is highly recommended** to seamlessly install the pre-compiled C-extensions via pip.

1. Set up a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate the virtual environment (Windows):
   ```bash
   .\.venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure `nltk`, `scikit-learn`, `pandas`, `gensim`, `matplotlib`, and `seaborn` are successfully installed).*

## 🚀 Running the Pipeline

To execute the full end-to-end evaluation suite (data ingestion, lemmatization, structural feature extraction, cross-validation, and hold-out testing), run:

```bash
python main.py
```

## 📊 Output Interpretations

The system automatically profiles the data and evaluates model performance. All generated visual artifacts are saved aggressively to the `outputs/` directory.

- **EDA Outputs**: 
  - `class_distribution.png` & `text_lengths.png`
  - `top_terms_ham.png` & `top_terms_spam.png`
- **Model Evaluation**:
  - `[model]_confusion_matrix.png` (Visualizes true vs false positive rates)
  - `[model]_pr_curve.png` (Precision-Recall tradeoff analysis for robust evaluation)
