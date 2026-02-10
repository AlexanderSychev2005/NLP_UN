import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from sklearn.metrics import mean_squared_error

data = pd.read_csv("monitoring_results.csv")


def parse_freqs(row):
    return [int(x) for x in row.split(", ")]


def parse_terms(row):
    return row.split(", ")


print("Generating Word Cloud...")

global_counter = {}
for index, row in data.iterrows():
    terms = parse_terms(row["Top_5_Terms"])
    freqs = parse_freqs(row["Frequencies"])

    for term, freq in zip(terms, freqs):
        term = term.lower().strip()
        global_counter[term] = global_counter.get(term, 0) + freq

wc = WordCloud(width=800, height=400, background_color="white")
wc.generate_from_frequencies(global_counter)

plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of News Terms (Week)")
plt.savefig("wordcloud_report.png")
plt.show()

print("Analyzing Global Top 3 Terms...")


top_3_global = sorted(global_counter.items(), key=lambda x: x[1], reverse=True)[:3]
top_3_words = [t[0] for t in top_3_global]
print(f"Top 3 persistent terms: {top_3_words}")


data["DateObject"] = pd.to_datetime(data["Date"])
unique_days = sorted(data["DateObject"].unique())
days_nums = np.arange(len(unique_days))

plt.figure(figsize=(12, 6))
for word in top_3_words:
    daily_freqs = []
    for day in unique_days:
        day_str = day.strftime("%Y-%m-%d")
        day_data = data[data["Date"] == day_str]

        word_sum = 0
        for index, row in day_data.iterrows():
            raw_terms = parse_terms(row["Top_5_Terms"])
            terms = [t.lower().strip() for t in raw_terms]

            freqs = parse_freqs(row["Frequencies"])
            if word in terms:
                idx = terms.index(word)
                word_sum += freqs[idx]
        daily_freqs.append(word_sum)

    x = days_nums
    y = np.array(daily_freqs)

    if len(x) > 1:  # Plot a trend line only if we have at least 2 data points (2 days)
        # y = kx + b
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        # MSE
        y_pred = m * x + c
        mse = mean_squared_error(y, y_pred)

        # Week forecast
        future_x = np.arange(len(x), len(x) + 7)
        future_y = m * future_x + c

        plt.plot(x, y, "o-", label=f"{word} (History)")
        plt.plot(future_x, future_y, "--", label=f"{word} (Forecast) MSE={mse:.2f}")

plt.title("Trend Analysis & Forecast for Top 3 Terms")
plt.xlabel("Days")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.savefig("trends_forecast_report.png")
plt.show()

print("Analysis complete. Charts saved.")
