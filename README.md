## What is gist

A small, real backend service that takes a long document (a news article, a support ticket, an incident report) and returns a short summary — served the way a production AI system actually gets served: batched, load-balanced across replicas, observed with real metrics, and autoscaled under realistic traffic.

## The problem it solves

Support desks, newsrooms, students, and ops teams all have the same problem: more long documents than anyone has time to read. "Summarize this, fast, at scale" is a real product category (Document AI, summarization endpoints) — not a synthetic benchmark.

## Architecture

```
Client
  |  sends a document
  v
API gateway         (routes to the least-loaded worker)
  |
  v
Batch queue         (batches by size or wait time)
  |
  v
Worker pool         (1-N model replicas)  <----  Autoscaler (watches load, adjusts pool size)
  |
  v
Response back to client (summary text)
```

Metrics (latency, queue depth, batch size, error rate) get collected from the queue and workers, feeding both the autoscaler and a Grafana dashboard.

## Tech stack

- **Model:** `sshleifer/distilbart-cnn-12-6` (distilled BART, fine-tuned for summarization, CPU-workable)
- **Data:** CNN/DailyMail dataset (`load_dataset("cnn_dailymail", "3.0.0")`) — real articles for realistic content and load testing
- **API:** Python + FastAPI
- **Containers:** Docker + Docker Compose (simulates multiple replicas locally)
- **Load testing:** Locust, with a scripted diurnal traffic shape
- **Metrics:** Prometheus + Grafana (swap for Cloud Monitoring if you deploy to GCP)
- **Optional deploy:** Any cloud provider

## Additional info

### Testing new capabilities

The throwaway_model_script.py is just that, a throwaway script meant to experiment with some specs (models, datasets, beams numbers, etc.).
To play around with it, make sure to have uv installed and run the following scripts:

**_Only on the first time_**, to create virtual environment, run:
`uv sync`

Then, to run the actual command run:
`uv run throwaway_model_script.py`

### Running worker locally

Get into the `server/worker` directory and execute: `uv run python -m app.main` 





