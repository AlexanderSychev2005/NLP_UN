import re

import spacy
import torch
from sentence_transformers import SentenceTransformer, util

print("Transformer loading...")
embedder = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)  # Supports both UK and EN


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


def extract_pos_tags(text):
    if re.search("[а-яА-ЯїЇєЄіІґҐ]", text):
        doc = nlp_uk(text)
    else:
        doc = nlp_en(text)
    return [(token.text, token.pos_) for token in doc]


corpus_texts = []
corpus_labels = []


for label, texts in dataset.items():
    for text in texts:
        corpus_texts.append(text)
        corpus_labels.append(label)

corpus_embeddings = embedder.encode(corpus_texts, convert_to_tensor=True)


def classify_text_semantic(input_text, threshold=0.35):
    """
    Classifies the text based on semantic similarity using Transformer. The threshold here is higher.
    """
    pos_tags = extract_pos_tags(input_text)

    # Vectotization
    query_embedding = embedder.encode(input_text, convert_to_tensor=True)

    # Cosine similarity calculation between the prompt and all documents in the corpus
    # Formula: similarity = cos(θ) = (A · B) / (||A|| * ||B||)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    # Looking for most similar document
    best_match_index = torch.argmax(cos_scores).item()
    best_match_score = cos_scores[best_match_index].item()

    if best_match_score >= threshold:
        predicted_category = corpus_labels[best_match_index]
    else:
        predicted_category = "Unknown"

    return predicted_category, best_match_score, pos_tags


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
        category, score, pos_tags = classify_text_semantic(phrase)
        print(f"Category: {category} (Confidence = {score:.2f})")
        print(
            f"Parts of the Speech: {[f'{word}[{pos}]' for word, pos in pos_tags[:5]]}..."
        )
        print("\n")
