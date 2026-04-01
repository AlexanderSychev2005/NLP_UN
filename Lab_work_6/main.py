import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
    nltk.download("punkt")


nlp = spacy.load("en_core_web_sm")
torch.manual_seed(42)

csv_file = "parsed_news.csv"
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"File {csv_file} not found!.")


news_df = pd.read_csv(csv_file)
print(f"Loaded {len(news_df)} news articles for further analysis.")

# NLP Processing
print("\nNLP processing (Tokenization, Lemmatization, POS)")
sample_text = news_df["text"].iloc[0]
print(f"Text: {sample_text}\n")

doc = nlp(sample_text)
stemmer = PorterStemmer()

print(f"{'Token':<15} | {'Lemma':<15} | {'Stem':<15} | {'Part of Speech'}")
print("-" * 60)
for token in doc:
    if not token.is_stop and not token.is_punct:
        print(
            f"{token.text:<15} | {token.lemma_:<15} | {stemmer.stem(token.text):<15} | {token.pos_}"
        )

print("\nNamed Entity Recognition (NER)")
print(f"{'Entity':<20} | {'Label':<15} | {'Description'}")
print("-" * 60)
for ent in doc.ents:
    description = spacy.explain(ent.label_)
    print(f"{ent.text:<20} | {ent.label_:<15} | {description}")


# Data preprocessing for deep Learning
def clean_text(text):
    return re.sub(r"[^a-zA-Z\s]", "", str(text)).lower().split()


# Vocabulary building
all_words = [word for text in news_df["text"] for word in clean_text(text)]
vocab = {
    word: i + 2 for i, (word, _) in enumerate(Counter(all_words).most_common(5000))
}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1


def encode_sentence(text, max_len=30):
    words = clean_text(text)
    encoded = [vocab.get(w, vocab["<UNK>"]) for w in words]
    if len(encoded) < max_len:
        encoded += [vocab["<PAD>"]] * (max_len - len(encoded))
    return encoded[:max_len]


X_encoded = np.array([encode_sentence(t) for t in news_df["text"]])


class TextClassifierANN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextClassifierANN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)  # Word-level pooling, simple average
        out = self.fc1(pooled)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def train_model(model, train_loader, criterion, optimizer, epochs=25):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


print("\nNeural Network for the sources classification")
y_source = LabelEncoder().fit_transform(news_df["source"].values)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_source, test_size=0.2, random_state=42
)

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long)
    ),
    batch_size=16,
    shuffle=True,
)
test_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)
    ),
    batch_size=16,
)

model_src = TextClassifierANN(len(vocab) + 2, 32, 3)
train_model(
    model_src,
    train_loader,
    nn.CrossEntropyLoss(),
    optim.Adam(model_src.parameters(), lr=0.005),
    epochs=30,
)
print(f"Accuracy (Sources): {evaluate_model(model_src, test_loader) * 100:.2f}%")

print("\nNeural Network for Sentiment Analysis")
analyzer = SentimentIntensityAnalyzer()


def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


news_df["sentiment"] = news_df["text"].apply(get_sentiment)
label_enc_sent = LabelEncoder()
y_sent = label_enc_sent.fit_transform(news_df["sentiment"].values)

X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(
    X_encoded, y_sent, test_size=0.2, random_state=42
)

train_loader_sent = DataLoader(
    TensorDataset(
        torch.tensor(X_train_sent, dtype=torch.long),
        torch.tensor(y_train_sent, dtype=torch.long),
    ),
    batch_size=16,
    shuffle=True,
)
test_loader_sent = DataLoader(
    TensorDataset(
        torch.tensor(X_test_sent, dtype=torch.long),
        torch.tensor(y_test_sent, dtype=torch.long),
    ),
    batch_size=16,
)

model_sent = TextClassifierANN(len(vocab) + 2, 32, 3)

train_model(
    model_sent,
    train_loader_sent,
    nn.CrossEntropyLoss(),
    optim.Adam(model_sent.parameters(), lr=0.005),
    epochs=30,
)

print(
    f"Accuracy (Sentiment): {evaluate_model(model_sent, test_loader_sent) * 100:.2f}%"
)

plt.figure(figsize=(7, 4))
ax = sns.countplot(
    data=news_df,
    x="sentiment",
    hue="sentiment",
    palette={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"},
    order=["Positive", "Neutral", "Negative"],
    legend=False,
)
plt.title("Sentiment Distribution of News Articles")
plt.xlabel("Sentiment")
plt.ylabel("Number of News Articles")

for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 7),
        textcoords="offset points",
    )
plt.tight_layout()
plt.show()
