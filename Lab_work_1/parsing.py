import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt

try:
    nltk.data.find("tokenizers/punkt_tab")
    nltk.data.find("corpora/stopwords")

except LookupError:
    nltk.download("punkt_tab")
    nltk.download("stopwords")


class NewsParser:
    def __init__(self):
        self.sources = {
            "BBC": "https://www.bbc.com/news",
            "CNBC": "https://www.cnbc.com/latest/",
        }
        self.csv_file = "monitoring_results.csv"
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
            "source",
            "copyright",
            "editor",
            "chief",
            "http",
            "https",
            "href",
        }
        self.stop_words.update(custom_stops)

        if not os.path.exists(self.csv_file):
            data = pd.DataFrame(
                columns=[
                    "Date",
                    "Time",
                    "Source",
                    "Top_5_Terms",
                    "Frequencies",
                    "Total_Freq_Sum",
                    "Comment",
                ]
            )
            data.to_csv(self.csv_file, index=False)

    def parse_page(self, url):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""

    def extract_news(self, html, source_name):
        soup = BeautifulSoup(html, "html.parser")
        text_content = ""

        for script in soup(["script", "style", "nav", "footer", "aside"]):
            script.extract()

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

        return text_content

    def process_text(self, text):
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters

        tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase

        filtered_tokens = [
            word for word in tokens if word not in self.stop_words and len(word) > 2
        ]  # Remove stopwords and short words
        return filtered_tokens

    def analyze_frequency(self, tokens):
        counter = Counter(tokens)
        top5 = counter.most_common(5)

        terms = [term[0] for term in top5]
        freqs = [term[1] for term in top5]
        total_sum = sum(freqs)

        return terms, freqs, total_sum

    def predict_trend(self):
        try:
            data = pd.read_csv(self.csv_file)
            if data.empty:
                print("There is no data yet.")
                return
            data["DateObject"] = pd.to_datetime(data["Date"])
            data["DayNumber"] = (data["DateObject"] - data["DateObject"].min()).dt.days

            x = data["DayNumber"].values
            y = data["Total_Freq_Sum"].values

            if len(set(x)) < 2:
                print("Not enough data points to predict trend.")
                return

            # Linear Regression, least squares, y = k*x + b

            A = np.vstack([x, np.ones(len(x))]).T
            k, b = np.linalg.lstsq(A, y, rcond=None)[0]

            print(f"Trend Line: y = {k:.2f}x + {b:.2f}")

            next_week_days = np.arange(x.max() + 1, x.max() + 8)
            predictions = k * next_week_days + b
            print("Predictions for next week:", np.round(predictions, 2))

            # Visualization
            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, color="blue", label="Actual Data")
            plt.plot(x, k * x + b, color="red", label="Trend Line")
            plt.plot(
                next_week_days,
                predictions,
                color="green",
                linestyle="--",
                label="Forecast",
            )
            plt.xlabel("Days from start")
            plt.ylabel("Total Frequency Sum")
            plt.title("Content Activity Trend & Forecast")
            plt.legend()
            plt.savefig("trend_forecast.png")
            print("Chart saved as trend_forecast.png")
        except Exception as e:
            print("Error predicting trend:", e)

    def monitor(self):
        print(f"Starting monitoring at {datetime.now()}")

        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        hour = datetime.now().hour

        if 6 <= hour < 12:
            time_of_day = "Morning"
        elif 12 <= hour < 18:
            time_of_day = "Day"
        else:
            time_of_day = "Evening"

        for source_name, url in self.sources.items():
            print(f"Parsing {source_name}...")
            html = self.parse_page(url)

            if html:
                raw_text = self.extract_news(html, source_name)
                cleaned_tokens = self.process_text(raw_text)

                if not cleaned_tokens:
                    print("No tokens found after processing.")
                    continue
                terms, freqs, total_sum = self.analyze_frequency(cleaned_tokens)

                new_row = {
                    "Date": current_date,
                    "Time": f"{time_of_day}: {current_time}",
                    "Source": source_name,
                    "Top_5_Terms": ", ".join(terms),
                    "Frequencies": ", ".join(map(str, freqs)),
                    "Total_Freq_Sum": total_sum,
                    "Comment": "",
                }
                data = pd.DataFrame([new_row])
                data.to_csv(self.csv_file, mode="a", header=False, index=False)
                print(f"Data saved for {source_name}")


if __name__ == "__main__":
    parser = NewsParser()
    # parser.monitor()
    parser.predict_trend()
