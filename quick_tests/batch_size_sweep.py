"""Model-only: summarizes the same Documents in batches of each size, one model.generate()
call per batch, to show how batch size alone changes throughput (no queue, no HTTP)."""

import argparse
import time

import torch

from common import GENERATE_KWARGS, RESULTS_DIR, CsvWriter, environment, load_documents, load_model, parse_ints, resolve_device

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sizes", type=parse_ints, default=[1, 2, 4, 8, 16, 32, 64], help="comma-separated batch sizes")
parser.add_argument("--beams", type=int, default=2, help="beam count (the worker's default is 2)")
parser.add_argument("--num-docs", type=int, default=256)
parser.add_argument("--out", default=RESULTS_DIR / "batch_size_sweep.csv")
args = parser.parse_args()

articles = [article for article, _ in load_documents(args.num_docs)]
device = resolve_device()
tokenizer, model = load_model(device)
writer = CsvWriter(args.out, environment())


def run_batch(batch: list[str]) -> float:
    input = tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(device)
    t1 = time.perf_counter()
    output = model.generate(**input, **GENERATE_KWARGS, num_beams=args.beams)
    if device == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    tokenizer.batch_decode(output, skip_special_tokens=True)
    return t2 - t1


throughputs = {}
for batch_size in args.sizes:
    batches = [articles[i:i + batch_size] for i in range(0, len(articles), batch_size)]
    try:
        run_batch(batches[0])  # warmup
        total = 0.0
        for batch_index, batch in enumerate(batches):
            latency = run_batch(batch)
            total += latency
            writer.write({"batch_size": batch_size, "num_beams": args.beams, "batch_index": batch_index,
                          "num_docs": len(batch), "latency_s": latency, "status": "ok"})
        throughputs[batch_size] = f"{len(articles) / total:.2f} docs/s"
    except torch.OutOfMemoryError:
        # Out of memory is a result too: record it and move on to the next size
        torch.cuda.empty_cache()
        writer.write({"batch_size": batch_size, "num_beams": args.beams, "batch_index": None,
                      "num_docs": None, "latency_s": None, "status": "oom"})
        throughputs[batch_size] = "out of memory"
    print(f"batch_size={batch_size}: {throughputs[batch_size]}")

writer.close()
print(f"\n{len(articles)} Documents on {device}, num_beams={args.beams}")
for batch_size, throughput in throughputs.items():
    print(f"{batch_size:>4}  {throughput}")
