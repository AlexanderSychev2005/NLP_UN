# Results Analysis
# 1. Stemming (NLTK):  is a faster, rule-based process that chops off word suffixes (often resulting in non-dictionary roots).
# e.g. stocks - stock, buying: buy. BUT universe - univers (not correct grammatically).

# 2. Lemmatization (spaCy): uses vocabulary and morphological analysis to return a valid dictionary word (lemma) based on context.
# e.g. bought - buy, better - good.

import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from collections import Counter
import re

nltk.download("punkt")
nltk.download("stopwords")

# NLTK stemming
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# SpaCy lemmatization
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Download the spacy module")
    exit()

try:
    dataset = pd.read_csv("monitoring_results_v2.csv")
except FileNotFoundError:
    print("First do monitoring in parsing.py!")
    exit()

print(f"Loaded {len(dataset)} records.")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


def nltk_stemming(text):
    text = clean_text(text)
    tokens = word_tokenize(text)

    # Filtering and stemming
    stems = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return stems


def spacy_lemma(text):
    doc = nlp(str(text).lower())  # Tokenization

    # Filtering and lemmatization if not stop word or punctuation
    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 2
    ]
    return lemmas


all_text_bbc = " ".join(dataset[dataset["Source"] == "BBC"]["Raw_Text"].dropna())
all_text_cnbc = " ".join(dataset[dataset["Source"] == "CNBC"]["Raw_Text"].dropna())

print("\n" + "=" * 20)
print("BBC News Analysis:")
print("=" * 20)

bbc_stems = nltk_stemming(all_text_bbc)
print(f"Top-5 (NLTK Stemming): {Counter(bbc_stems).most_common(5)}")

bbc_lemmas = spacy_lemma(all_text_bbc)
print(f"Top-5 (spaCy Lemmatization): {Counter(bbc_lemmas).most_common(5)}")

print("\n" + "=" * 20)
print("CNBC News Analysis:")
print("=" * 20)

cnbc_stems = nltk_stemming(all_text_cnbc)
print(f"Top-5 (NLTK Stemming): {Counter(cnbc_stems).most_common(5)}")

cnbc_lemmas = spacy_lemma(all_text_cnbc)
print(f"Top-5 (spaCy Lemmatization): {Counter(cnbc_lemmas).most_common(5)}")
