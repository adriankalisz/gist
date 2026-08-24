import requests
import time
from datasets import load_dataset

num_articles = 10

articles = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"test[:{num_articles}]")
times = [0] * num_articles

for i in range(num_articles):
    article = articles[i]["article"]
    t1 = time.perf_counter()
    response = requests.post("http://localhost:8000/summarize", json={"text": article})
    t2 = time.perf_counter()
    times[i] = t2 - t1
print(f"Total time for {num_articles} requests: {sum(times):.2f} seconds")
print(f"Average inference time for {num_articles} requests: {sum(times) / num_articles:.2f} seconds")
print(f"Specific inference times for each request: {times}")
