import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
import os
import re
from datetime import datetime


try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")


class NewsMonitor:
    def __init__(self):
        self.sources = {
            "BBC": "https://www.bbc.com/news",
            "CNBC": "https://www.cnbc.com/latest/",
        }

        self.csv_file = "monitoring_results_v2.csv"
        self.stop_words = set(stopwords.words("english"))

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
            "live",
            "update",
            "read",
            "more",
            "watch",
            "video",
            "full",
            "story",
            "bbc",
            "cnbc",
            "london",
            "image",
            "source",
            "page",
            "one",
            "two",
            "first",
            "last",
            "make",
            "get",
            "take",
            "use",
            "could",
            "may",
            "would",
            "also",
            "copyright",
            "rights",
            "http",
        }
        self.stop_words.update(custom_stops)

        if not os.path.exists(self.csv_file):
            columns = [
                "Date",
                "Time",
                "Source",
                "Raw_Text",
                "Top_5_Terms",
                "Frequencies",
            ]
            dataset = pd.DataFrame(columns=columns)
            dataset.to_csv(self.csv_file, index=False)

    def get_news(self, url):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error parsing news from {url}: {e}")
            return None

    def extract_text(self, html, source_name):
        soup = BeautifulSoup(html, "html.parser")
        text_content = ""

        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.extract()

        if source_name == "BBC":
            headlines = soup.find_all(attrs={"data-testid": "card-headline"})
            for h in headlines:
                text_content += h.get_text(separator=" ") + " "

            descriptions = soup.find_all(attrs={"data-testid": "card-description"})
            for d in descriptions:
                text_content += d.get_text(separator=" ") + " "

        elif source_name == "CNBC":
            titles = soup.find_all(class_="Card-title")
            for t in titles:
                if len(t.text) > 20:
                    text_content += t.get_text(separator=" ") + " "

        return text_content.strip()

    def process_text(self, text):
        clean_text = re.sub(r"[^a-zA-Z\s]", "", text)
        tokens = word_tokenize(clean_text.lower())
        filtered = [w for w in tokens if w not in self.stop_words and len(w) > 2]
        return filtered

    def analyze_and_save(self):
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M")

        h = datetime.now().hour
        time_of_day = "Morning" if 6 <= h < 12 else "Day" if 12 <= h < 18 else "Evening"

        print(f"Monitoring start: {current_date}, {current_time}: {time_of_day}")

        for name, url in self.sources.items():
            print(f"Parsing {name}...")
            html = self.get_news(url)

            if html:
                raw_text = self.extract_text(html, name)

                tokens = self.process_text(raw_text)
                if not tokens:
                    print("No content found")
                    continue

                counter = Counter(tokens)
                top_5 = counter.most_common(5)
                terms = [t[0] for t in top_5]
                freqs = [t[1] for t in top_5]

                raw_text = (
                    raw_text.replace("\n", " ").replace("\r", "").replace(",", ";")
                )

                row = {
                    "Date": current_date,
                    "Time": f"{time_of_day}: {current_time}",
                    "Source": name,
                    "Raw_Text": raw_text,
                    "Top_5_Terms": ", ".join(terms),
                    "Frequencies": ", ".join(map(str, freqs)),
                }
                dataset = pd.DataFrame([row])
                dataset.to_csv(self.csv_file, mode="a", header=False, index=False)

            else:
                print(f"Failed fething {name}: {url}")


if __name__ == "__main__":
    parser = NewsMonitor()
    parser.analyze_and_save()
