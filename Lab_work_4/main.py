import re
from collections import Counter
from typing import List

import nltk
import requests
import seaborn as sns
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.util import bigrams

# ML
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Download necessary NLTK data
try:
    nltk.data.find("tokenizers/punkt_tab")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    nltk.download("punkt")


class NewsParserNLP:
    def __init__(self) -> None:
        self.sources = {
            "BBC": "https://www.bbc.com/news",
            "CNBC": "https://www.cnbc.com/latest/",
        }
        self.stop_words = set(stopwords.words("english"))

        # Custom stop words to filter out common journalistic noise
        custom_stops = {
            "said",
            "says",
            "say",
            "news",
            "report",
            "world",
            "year",
            "years",
            "time",
            "day",
            "week",
            "today",
            "new",
            "bbc",
            "cnbc",
            "image",
            "video",
            "watch",
            "read",
            "more",
            "found",
            "first",
            "two",
            "one",
            "make",
            "take",
            "get",
            "use",
            "could",
            "may",
            "would",
            "also",
        }
        self.stop_words.update(custom_stops)

    def parse_page(self, url: str) -> str:
        """Fetches the HTML content of the given URL."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""

    def extract_news_list(self, html: str, source_name: str) -> List[str]:
        """Parses HTML and extracts a list of news headlines/titles."""

        soup = BeautifulSoup(html, "html.parser")
        news_list = []
        for script in soup(["script", "style", "nav", "footer", "aside"]):
            script.extract()

        # Remove unwanted tags
        if source_name == "BBC":
            headlines = soup.find_all(attrs={"data-testid": "card-headline"})
            for h in headlines:
                text = h.get_text(separator=" ").strip()
                if len(text) > 20:
                    news_list.append(text)

        elif source_name == "CNBC":
            titles = soup.find_all(class_="Card-title")
            for t in titles:
                text = t.get_text(separator=" ").strip()
                if len(text) > 20:
                    news_list.append(text)

        return news_list

    def process_tokenization_text(self, text: str) -> List[str]:
        """Cleans the text and tokenizes it, removing stop words and short words."""

        text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation and numbers
        tokens = nltk.word_tokenize(text.lower())
        filtered_tokens = [w for w in tokens if w not in self.stop_words and len(w) > 2]
        return filtered_tokens

    def get_all_news(self) -> List[str]:
        """Iterates over all defined sources, fetches and aggregates the news."""
        all_news = []

        for source_name, url in self.sources.items():
            print(f"Parsing {source_name}...")
            html = self.parse_page(url)

            if html:
                news = self.extract_news_list(html, source_name)
                all_news.extend(news)
                print(f"Got {len(news)} news from {source_name}")

        return list(set(all_news))  # Return unique items only


if __name__ == "__main__":
    parser = NewsParserNLP()
    scraped_news = parser.get_all_news()

    print(f"\nWe have {len(scraped_news)} scraped news articles for further analysis\n")

    # Level 1 - Supervised Learning - Classification
    print("Level 1 - Supervised Learning - Classification")

    categories = [
        "talk.politics.misc",  # General politics
        "talk.politics.mideast",  # Middle East politics
        "misc.forsale",  # Proxy for sales/markets/business
        "sci.space",  # Space/Tech
        "sci.med",  # Health/Medical news
        "comp.sys.mac.hardware",  # Proxy for general tech news
    ]
    print("Loading the dataset (20 Newsgroups)...")

    newsgroup_data = fetch_20newsgroups(subset="train", categories=categories)

    # Split data into training and testing sets
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        newsgroup_data.data, newsgroup_data.target, test_size=0.2, random_state=42
    )

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    # Train a Logistic Regression model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Evaluate model
    acc = accuracy_score(y_test, y_pred)
    print(f"Classification Accuracy: {acc * 100:.2f}%\n")

    print("\nDetailed Classification Report:")
    print(
        classification_report(y_test, y_pred, target_names=newsgroup_data.target_names)
    )
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=newsgroup_data.target_names,
        yticklabels=newsgroup_data.target_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Predict categories for our scraped news
    X_scraped = vectorizer.transform(scraped_news)
    predictions = clf.predict(X_scraped)

    print("Examples:")
    for i in range(min(5, len(scraped_news))):
        cat_name = newsgroup_data.target_names[predictions[i]]
        print(f"[{cat_name.upper()}] - {scraped_news[i]}")

    # Level 2 - Frequency and Probability Analysis
    print("Level 2 - Frequency and Probability Analysis")

    full_text = " ".join(scraped_news)
    all_tokens = parser.process_tokenization_text(full_text)

    # 1. Bigram Analysis
    bigrams_freq = Counter(list(bigrams(all_tokens)))
    print("Top-5 bigrams:")
    for bg, freq in bigrams_freq.most_common(5):
        print(f"{bg[0]} {bg[1]}: {freq} occurrences")

    # 2. Words length distribution
    word_lengths = [len(w) for w in all_tokens]
    plt.figure(figsize=(8, 4))
    sns.histplot(word_lengths, bins=15, color="purple", kde=True)
    plt.title("Words length distribution")
    plt.xlabel("Word length (characters)")
    plt.ylabel("Frequency")
    plt.show()

    # 3. Lexical dispersion
    top_3_words = [word for word, count in Counter(all_tokens).most_common(3)]
    text_obj = nltk.Text(all_tokens)
    text_obj.dispersion_plot(top_3_words)
    plt.title("Lexical Dispersion for Top Words")
    plt.show()

    # Level 3 - Unsupervised Learning (K-Means Clustering)
    print("Level 3 -Clustering (Unsupervised Learning)")

    # Re-vectorize specifically for clustering using our custom stop words
    custom_vectorizer = TfidfVectorizer(stop_words=list(parser.stop_words))
    X_scraped_tfidf = custom_vectorizer.fit_transform(scraped_news)

    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scraped_tfidf)  # Train and get labels

    # Extract top terms per cluster
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = custom_vectorizer.get_feature_names_out()

    for i in range(num_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :7]]
        print(f"Cluster {i}: {', '.join(top_terms)}")

    print("\nGenerating Cluster Visualization...")
    # Use TruncatedSVD to reduce dimensionality of the sparse TF-IDF matrix to 2D
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_scraped_tfidf)

    plt.figure(figsize=(8, 6))
    # Create a scatter plot colored by K-Means cluster labels
    sns.scatterplot(
        x=X_2d[:, 0], y=X_2d[:, 1], hue=kmeans_labels, palette="viridis", s=100
    )
    plt.title("K-Means Clustering of News Headlines (2D Projection)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Cluster")
    plt.show()
