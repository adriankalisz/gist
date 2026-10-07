import argparse
import requests
import time

from common import RESULTS_DIR, CsvWriter, environment, load_documents

# Sends Documents to a running worker one at a time (no concurrency)
parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8000/summarize")
parser.add_argument("--num-docs", type=int, default=10)
parser.add_argument("--label", default="local", help="names the worker setup under test")
parser.add_argument("--worker-commit", help="commit the worker runs (default: this checkout's)")
parser.add_argument("--out", help="CSV path (default: results/ten_requests_<label>.csv)")
args = parser.parse_args()

num_articles = args.num_docs
articles = load_documents(num_articles)
writer = CsvWriter(args.out or RESULTS_DIR / f"ten_requests_{args.label}.csv", environment(args.worker_commit))
times = [0] * num_articles
model_times = [0] * num_articles

for i in range(num_articles):
    article = articles[i][0]
    t1 = time.perf_counter()
    response = requests.post(args.url, json={"text": article})
    t2 = time.perf_counter()
    times[i] = t2 - t1
    model_times[i] = response.json().get("inference_time", 0)
    writer.write({"label": args.label, "doc_index": i, "status_code": response.status_code,
                  "latency_s": times[i], "inference_time": model_times[i]})

writer.close()
print(f"Total time for {num_articles} requests: {sum(times):.2f} seconds")
print(f"Average post request time for {num_articles} requests: {sum(times) / num_articles:.2f} seconds")
print(f"Specific post request times for each request: {times}")
print(f"Model times for each request inference: {model_times}")
