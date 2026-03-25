import warnings
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from transformers import pipeline

# Download VADER lexicon if not already present
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon")


class NewsAnalyzer:
    """
    A class to analyze news texts, including text similarity between sources
    and sentiment analysis.
    """

    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        try:
            # Load the complete dataset for sentiment analysis
            self.df_full = pd.read_csv(self.csv_file)

            if self.df_full is not None and not self.df_full.empty:
                self.df_latest = self.df_full.drop_duplicates(
                    subset=["Source"], keep="last"
                ).reset_index(drop=True)
        except FileNotFoundError:
            print(f"File {self.csv_file} not found")
            self.df_full = None
            self.df_latest = None

    def analyze_similarity(self) -> None:
        """
        Performs comparative similarity analysis using TF-IDF and Cosine Similarity.
        Uses only the latest news items to prevent matrix clutter.
        Saves the resulting heatmap as an image.
        """
        print("\n I Level: Similarity analysis (Cosine Similarity)")

        if self.df_latest is None or self.df_latest.empty:
            print("No data available for similarity analysis.")
            return

        texts = self.df_latest["Raw_Text"].fillna("").tolist()
        sources = self.df_latest["Source"].tolist()

        # Texts -> Vectors
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                sim_score = similarity_matrix[i][j]
                print(
                    f"Similarity between {sources[i]} and {sources[j]}: {sim_score:.4f} ({sim_score * 100:.1f}%)"
                )

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            similarity_matrix,
            annot=True,
            cmap="YlGnBu",
            xticklabels=sources,
            yticklabels=sources,
            fmt=".2f",
        )
        plt.title("Similarity matrix of news (Cosine Similarity)")
        plt.tight_layout()
        plt.savefig("similarity_heatmap.png")
        print("Similarity plot was saved as 'similarity_heatmap.png'")

    def analyze_sentiment(self) -> None:
        """
        Performs sentence-level Sentiment Analysis comparing three models:
        VADER (Lexicon), TextBlob (Heuristics), and Hugging Face (Neural Network).
        """
        print("\nII level: Multi-Model Sentiment Analysis (Sentence Level)")

        if self.df_full is None or self.df_full.empty:
            print("No data available for sentiment analysis.")
            return

        sia = SentimentIntensityAnalyzer()

        # Suppress warnings for cleaner console output
        warnings.filterwarnings("ignore")
        print("HLoading Hugging Face DistilBERT model...")
        hf_analyzer = pipeline("sentiment-analysis", truncation=True, max_length=512)

        comparison_results = []

        for index, row in self.df_full.iterrows():
            text = str(row["Raw_Text"])
            source = row["Source"]

            # Tokenize the large text into individual sentences
            sentences = sent_tokenize(text)

            if not sentences:
                continue

            vader_scores = []
            textblob_scores = []
            hf_scores = []

            print(f"Analysis of source: {source} ({len(sentences)} sentences)...")

            for sent in sentences:
                if not sent.strip():
                    continue

                # 1. VADER
                vader_scores.append(sia.polarity_scores(sent)["compound"])

                # 2. TextBlob
                textblob_scores.append(TextBlob(sent).sentiment.polarity)

                # 3. Hugging Face
                try:
                    hf_res = hf_analyzer(sent)[0]
                    # Map POSITIVE/NEGATIVE labels to a -1.0 to 1.0 scale
                    hf_val = (
                        hf_res["score"]
                        if hf_res["label"] == "POSITIVE"
                        else -hf_res["score"]
                    )
                    hf_scores.append(hf_val)
                except Exception:
                    hf_scores.append(0.0)

            # Calculate the mean score for the source across all its sentences
            avg_vader = sum(vader_scores) / len(vader_scores) if vader_scores else 0
            avg_textblob = (
                sum(textblob_scores) / len(textblob_scores) if textblob_scores else 0
            )
            avg_hf = sum(hf_scores) / len(hf_scores) if hf_scores else 0

            print(
                f"  VADER: {avg_vader:.4f} | TextBlob: {avg_textblob:.4f} | Hugging Face: {avg_hf:.4f}\n"
            )

            comparison_results.append(
                {
                    "Source": source,
                    "VADER": avg_vader,
                    "TextBlob": avg_textblob,
                    "Hugging Face": avg_hf,
                }
            )

        # Prepare data for plotting (Melting the DataFrame for Seaborn grouped barplot)
        results_df = pd.DataFrame(comparison_results)
        melted_df = results_df.melt(
            id_vars=["Source"], var_name="Model", value_name="Sentiment Score"
        )

        # Visualizing the Multi-Model Comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="Source",
            y="Sentiment Score",
            hue="Model",
            data=melted_df,
            palette="viridis",
        )
        plt.title("Sentiment Analysis Algorithm Comparison (Sentence Average)")
        plt.ylabel("Average Sentiment Score (-1.0 to 1.0)")
        plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
        plt.legend(title="NLP Model")
        plt.tight_layout()

        plt.savefig("multi_model_sentiment.png")
        print("Comparison plot saved as 'multi_model_sentiment.png'")

    def analyze_ner(self) -> None:
        """
        Performs Named Entity Recognition (NER) to extract key entities (Organizations,
        People, Locations) using both spaCy and Hugging Face pipelines.
        """
        print("\nAdditional Level: Named Entity Recognition (spaCy vs Hugging Face)")

        if self.df_latest is None or self.df_latest.empty:
            print("No data available for NER analysis.")
            return

        warnings.filterwarnings("ignore")

        # Load spaCy Model
        print("Loading spaCy model (en_core_web_sm)...")
        try:
            nlp_spacy = spacy.load("en_core_web_sm")
        except OSError:
            print(
                "Error: spaCy model not found. Please run: python -m spacy download en_core_web_sm"
            )
            return

        # Load Hugging Face NER Model
        # aggregation_strategy="simple" merges sub-tokens (e.g., "New", "York" -> "New York")
        print(
            "Loading Hugging Face NER model (dbmdz/bert-large-cased-finetuned-conll03-english)..."
        )
        hf_ner = pipeline("ner", aggregation_strategy="simple")

        for index, row in self.df_latest.iterrows():
            source = row["Source"]
            raw_text = str(row["Raw_Text"])

            # Truncate text for Hugging Face to avoid severe performance drops and token limits
            truncated_text = raw_text[:1500]

            print(f"\nTop Entities for {source}:")

            # 1. spaCy NER
            doc = nlp_spacy(raw_text)
            # Filter for specific entity types (ORG, PERSON, GPE - Geopolitical Entity)
            spacy_entities = [
                ent.text.strip()
                for ent in doc.ents
                if ent.label_ in ["ORG", "PERSON", "GPE", "LOC"]
                and len(ent.text.strip()) > 2
            ]
            spacy_top_5 = Counter(spacy_entities).most_common(5)

            # 2. Hugging Face NER
            try:
                hf_results = hf_ner(truncated_text)
                # Filter out generic words, keep only ORG, PER, LOC
                hf_entities = [
                    ent["word"].strip()
                    for ent in hf_results
                    if ent["entity_group"] in ["ORG", "PER", "LOC"]
                    and len(ent["word"].strip()) > 2
                ]
                hf_top_5 = Counter(hf_entities).most_common(5)
            except Exception as e:
                print(f"Hugging Face NER failed for {source}: {e}")
                hf_top_5 = []

            print(
                f"  [spaCy Top 5]: {', '.join([f'{word} ({count})' for word, count in spacy_top_5])}"
            )
            print(
                f"  [HF Top 5]:    {', '.join([f'{word} ({count})' for word, count in hf_top_5])}"
            )


if __name__ == "__main__":
    analyzer = NewsAnalyzer("monitoring_results_v3.csv")
    analyzer.analyze_similarity()
    analyzer.analyze_sentiment()
    analyzer.analyze_ner()
