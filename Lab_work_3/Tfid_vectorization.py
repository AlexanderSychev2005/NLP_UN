import re

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nlp_uk = spacy.load("uk_core_news_sm")
    nlp_en = spacy.load("en_core_web_sm")
except OSError:
    print("Error: SpaCy were not found.")
    print(
        "Run: python -m spacy download uk_core_news_sm and python -m spacy download en_core_web_sm"
    )
    exit()


dataset = {
    "Радіоелектроніка": [
        "Принципова електрична схема підсилювача на транзисторах і резисторах.",
        "Паяння мікросхем, конденсаторів та плат паяльником.",
        "Soldering microchips, capacitors and resistors on the printed circuit board.",
        "Analyzing high radio frequency antenna amplifier signals with an oscilloscope.",
    ],
    "Програмування": [
        "Розробка алгоритмів машинного навчання, написання коду на Python.",
        "Створення об'єктно-орієнтованої архітектури баз даних.",
        "Writing a Python script, debugging code and compiling algorithms.",
        "Deploying web servers, client interfaces and software frameworks.",
    ],
    "Машинобудування": [
        "Проектування деталей двигуна внутрішнього згоряння та турбін.",
        "Кінематичний аналіз зубчастої передачі, валів та підшипників.",
        "Thermodynamic analysis of a combustion engine and mechanical gears.",
        "Hydraulics, pneumatics and welding of metal heavy mechanisms.",
    ],
}


def process_text(text):
    """
    Detects language (Ukrainian or English), performs POS tagging and saves only semantically
    important words (nouns, verbs, adjectives)
    """
    if re.search("[а-яА-ЯїЇєЄіІґҐ]", text):  # Checking for cyrillic
        doc = nlp_uk(text)
    else:
        doc = nlp_en(text)

    filtered_tokens = []
    pos_tags = []
    for token in doc:
        pos_tags.append((token.text, token.pos_))
        if token.pos_ in [
            "NOUN",
            "PROPN",
            "ADJ",
            "VERB",
        ]:  # PROPN - Proper Noun (Kyiv, Byzantium)
            filtered_tokens.append(token.lemma_.lower())

    return " ".join(filtered_tokens), pos_tags


corpus = []
labels = []


for label, texts in dataset.items():
    for text in texts:
        filtered_text, _ = process_text(text)
        corpus.append(filtered_text)
        labels.append(label)

vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(corpus)


def classify_text(input_text, threshold=0.15):
    """
    Performs the text vectorizing and compare it with the existing corpus.
    If the confidence < threshold, unknown topic.
    """
    filtered_input, tags = process_text(input_text)
    if not filtered_input:
        return "Unknown", 0.0, tags

    # Text vectorization
    input_vector = vectorizer.transform([filtered_input])

    # Calculates the cosine similarity for all documents in the corpus
    similarities = cosine_similarity(input_vector, X_vectors)[0]

    max_index = similarities.argmax()
    max_score = similarities[max_index]

    if max_score >= threshold:
        predicted_category = labels[max_index]
    else:
        predicted_category = "Unknown"

    return predicted_category, max_score, tags


if __name__ == "__main__":
    test_phrases = [
        "Нам потрібно припаяти цей транзистор до плати.",  # Radio electronics
        "I need to debug this Python code and fix the database schema.",  # Programming
        "Інженер розробив нове креслення зубчастого валу для двигуна.",  # Mechanical engineering
        "Завтра ми йдемо в кіно на новий фільм.",  # Unknown (noise)
        "The thermodynamics of the new turbine mechanism are complex.",  # Mechanical Engineering (EN)
    ]

    for phrase in test_phrases:
        print(f"Input text: '{phrase}'")
        category, score, pos_tags = classify_text(phrase)
        print(f"Category: {category} (Confidence = {score:.2f})")
        print(
            f"Parts of the Speech: {[f'{word}[{pos}]' for word, pos in pos_tags[:5]]}..."
        )
        print("\n")
