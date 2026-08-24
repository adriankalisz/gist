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

### Running the app locally

Since the app is fully containerized, user needs Docker installed. The steps are:

1. Get into the `server/worker` (`cd server/worker`).
2. If this is the **first time** launching the app, run `docker build -t gist-worker .` (builds the Docker image)
3. Run: `docker run -p 8000:8000 gist-worker` (mapps the docker's port 8000 to user's port 8000 and runs the image)

 

### Testing new capabilities

Inside `quick_tests` directory, are the experiments measuring the quality and latencies for response and different components of the app. In order to run them, launch the app and run the following commands

**_Only on the first time_**, to create virtual environment, run:
`uv sync`

Then, to run the actual command run:
`uv run quick/tests[SCRIPT_NAME].py`





