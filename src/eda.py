import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from pathlib import Path

# Set the visual style for the plots
sns.set_theme(style="whitegrid")

def plot_class_distribution(df, output_path="outputs/class_distribution.png"):
    """
    Generate a bar chart showing the distribution of Spam vs. Ham.
    """
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x='label', hue='label', palette='viridis', legend=False)
    plt.title("Class Distribution (0: Ham, 1: Spam)", fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(ticks=[0, 1], labels=["Ham", "Spam"])
    
    # Add counts on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_text_lengths(df, output_path="outputs/text_lengths.png"):
    """
    Generate a histogram comparing text lengths (word counts) of spam vs. ham.
    """
    df = df.copy()
    df['word_count'] = df['text'].str.split().str.len()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='word_count', hue='label', kde=True, bins=50, palette='magma', element="step")
    plt.title("Text Length Distribution (Word Count) by Class", fontsize=14)
    plt.xlabel("Number of Words", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    # Zoom in on shorter messages if there are outliers, but let's see. 
    # Usually emails vary a lot, so we might want to clip or show common range.
    plt.xlim(0, df['word_count'].quantile(0.95)) 
    plt.legend(title="Class", labels=["Spam", "Ham"])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_most_frequent_terms(df, output_path_prefix="outputs/top_terms"):
    """
    Generate bar charts of the most frequent terms for each class.
    """
    for label, label_name in [(0, "Ham"), (1, "Spam")]:
        subset = df[df['label'] == label]['text']
        all_words = " ".join(subset).split()
        
        # Simple stopword list to make it cleaner
        common_stopwords = set(['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'i', 'for', 'you', 'on', 'this', 'that', 'with', 'be', 'are', 'as', 'at', 'if', 'we', 'have', 'your', 'was', 'not', 'but', 'by', 'me', 'from', 'my', 'or', 'an'])
        filtered_words = [w for w in all_words if w not in common_stopwords and len(w) > 2]
        
        words_count = Counter(filtered_words).most_common(20)
        words_df = pd.DataFrame(words_count, columns=['word', 'count'])
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=words_df, x='count', y='word', hue='word', palette='coolwarm', legend=False)
        plt.title(f"Top 20 Frequent Terms in {label_name} Emails", fontsize=14)
        plt.xlabel("Frequency", fontsize=12)
        plt.ylabel("Term", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{output_path_prefix}_{label_name.lower()}.png")
        plt.close()

def run_eda(df):
    """
    Run all EDA functions and save plots to the outputs directory.
    """
    # Ensure outputs directory exists
    Path("outputs").mkdir(exist_ok=True)
    
    print("Generating EDA plots...")
    plot_class_distribution(df)
    plot_text_lengths(df)
    plot_most_frequent_terms(df)
    print("EDA plots saved in 'outputs/' directory.")
