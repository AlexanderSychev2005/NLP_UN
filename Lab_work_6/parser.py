import pandas as pd
import requests
from bs4 import BeautifulSoup


class NewsScraper:
    def __init__(self):
        self.sources: dict[str, str] = {
            "BBC": "https://www.bbc.com/news",
            "CNBC": "https://www.cnbc.com/latest/",
            "Al Jazeera": "https://www.aljazeera.com/news/",
        }
        self.filename = "parsed_news.csv"

    def fetch_html(self, url: str) -> str:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()
            return res.text
        except Exception as e:
            print(f"Error parsing news from {url}: {e}")
            return ""

    def scrape_all(self) -> None:
        data = []

        # BBC
        print("BBC Parsing...")
        html = self.fetch_html(self.sources["BBC"])
        if html:
            soup = BeautifulSoup(html, "html.parser")
            for h in soup.find_all(attrs={"data-testid": "card-headline"}):
                if len(h.text) > 20:
                    data.append({"text": h.text.strip(), "source": "BBC"})

        # CNBC
        print("CNBC Parsing...")
        html = self.fetch_html(self.sources["CNBC"])
        if html:
            soup = BeautifulSoup(html, "html.parser")
            for t in soup.find_all(class_="Card-title"):
                if len(t.text) > 20:
                    data.append({"text": t.text.strip(), "source": "CNBC"})

        # Al Jazeera
        print("Al Jazeera Parsing...")
        html = self.fetch_html(self.sources["Al Jazeera"])
        if html:
            soup = BeautifulSoup(html, "html.parser")
            for h3 in soup.find_all(class_="article-card__title"):
                text = h3.get_text().strip()
                if len(text) > 20:
                    data.append({"text": text, "source": "Al Jazeera"})

        dataset = pd.DataFrame(data).drop_duplicates(subset=["text"])
        dataset.to_csv(self.filename, index=False)
        print(f"\nSuccess! Collected unique news articles: {len(dataset)}.")
        print(f"Saved to {self.filename}.")


if __name__ == "__main__":
    scraper = NewsScraper()
    scraper.scrape_all()
