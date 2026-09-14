import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from datasets import load_dataset

num_articles = 10
dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"test[:{num_articles}]")
articles = [dataset[i]["article"] for i in range(num_articles)]


def send_request(index, article):
    t1 = time.perf_counter()
    response = requests.post("http://localhost:8000/summarize", json={"text": article})
    t2 = time.perf_counter()
    data = response.json()
    return {
        "index": index,
        "status_code": response.status_code,
        "wall_time": t2 - t1,
        "inference_time": data.get("inference_time", 0),
        "summary": data.get("summary", ""),
    }


results = [None] * num_articles
t_start = time.perf_counter()
with ThreadPoolExecutor(max_workers=num_articles) as executor:
    future_to_index = {executor.submit(send_request, i, articles[i]): i for i in range(num_articles)}
    for fut in as_completed(future_to_index):
        result = fut.result()
        results[result["index"]] = result
t_end = time.perf_counter()

wall_times = [r["wall_time"] for r in results]
inference_times = [r["inference_time"] for r in results]
summaries = [r["summary"] for r in results]

print(f"Total wall-clock time for {num_articles} concurrent requests: {t_end - t_start:.2f} seconds")
print(f"Average per-request wall time: {sum(wall_times) / num_articles:.2f} seconds")
print(f"Per-request wall times: {wall_times}")
print(f"Model inference times (per request): {inference_times}")
print(f"Status codes: {[r['status_code'] for r in results]}")

# Several requests sharing an identical inference_time is the expected, correct
# signature of real batching (one shared model.generate() call) - not a bug.
# Only duplicate/identical summary text across different articles would
# indicate the batch-mixing bug.
unique_summaries = len(set(summaries))
print(f"Unique summaries among {num_articles} responses: {unique_summaries}/{num_articles}")
if unique_summaries < num_articles:
    print("WARNING: duplicate/identical summaries detected across different articles - possible batch-mixing bug!")
else:
    print("OK: all summaries are distinct - no evidence of the batch-mixing bug.")
