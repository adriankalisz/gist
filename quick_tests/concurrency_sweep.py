"""End-to-end: at each concurrency level, that many clients share one queue of Documents and
each sends its next request as soon as its previous one returns (closed loop). Records
every request's latency, from the moment it is sent, against a running worker."""

import argparse
import queue
import threading
import time

import requests

from common import RESULTS_DIR, CsvWriter, environment, load_documents, parse_ints, percentile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://localhost:8000/summarize")
parser.add_argument("--clients", type=parse_ints, default=[1, 8, 32, 128], help="comma-separated concurrency levels")
parser.add_argument("--num-docs", type=int, default=256, help="Documents sent per concurrency level")
parser.add_argument("--label", default="local", help="names the worker setup under test, e.g. baseline, bs8")
parser.add_argument("--worker-commit", help="commit the worker runs (default: this checkout's)")
parser.add_argument("--out", help="CSV path (default: results/concurrency_sweep_<label>.csv)")
args = parser.parse_args()

articles = [article for article, _ in load_documents(args.num_docs)]
writer = CsvWriter(args.out or RESULTS_DIR / f"concurrency_sweep_{args.label}.csv", environment(args.worker_commit))
write_lock = threading.Lock()


def client(client_index: int, level: int, todo: queue.Queue, start: float, rows: list):
    session = requests.Session()
    while True:
        try:
            doc_index = todo.get_nowait()
        except queue.Empty:
            return
        sent = time.perf_counter()
        response = session.post(args.url, json={"text": articles[doc_index]}, timeout=600)
        latency = time.perf_counter() - sent
        data = response.json() if response.ok else {}
        row = {"label": args.label, "clients": level, "client": client_index, "doc_index": doc_index,
               "status_code": response.status_code, "sent_at_s": sent - start, "latency_s": latency,
               "inference_time": data.get("inference_time"), "summary": data.get("summary")}
        with write_lock:
            writer.write(row)
            rows.append(row)


for level in args.clients:
    todo = queue.Queue()
    for doc_index in range(len(articles)):
        todo.put(doc_index)
    rows = []
    start = time.perf_counter()
    threads = [threading.Thread(target=client, args=(i, level, todo, start, rows)) for i in range(level)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - start

    ok = [r for r in rows if r["status_code"] == 200]
    latencies = [r["latency_s"] for r in ok]
    print(f"clients={level:>3}  ok={len(ok)}/{len(rows)}  throughput={len(ok) / elapsed:.2f} docs/s  "
          f"p50={percentile(latencies, 50):.2f}s  p95={percentile(latencies, 95):.2f}s" if ok else
          f"clients={level:>3}  ok=0/{len(rows)}")
    # Requests in one Batch share an inference_time, which is expected. Identical summaries for
    # different Documents would mean Batch results were handed back to the wrong requests.
    if len({r["summary"] for r in ok}) < len(ok):
        print("  WARNING: duplicate summaries across different Documents - possible batch-mixing bug!")

writer.close()
