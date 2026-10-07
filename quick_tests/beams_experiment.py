"""Model-only: summarizes each Document one at a time at every beam count, recording
latency and ROUGE against the human highlights, to pick the worker's beam count."""

import argparse
import time

import torch
from rouge_score import rouge_scorer

from common import GENERATE_KWARGS, MAX_DOCUMENT_CHARS, RESULTS_DIR, CsvWriter, environment, load_documents, load_model, parse_ints, resolve_device

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--beams", type=parse_ints, default=[1, 2, 4, 6], help="comma-separated beam counts")
parser.add_argument("--num-docs", type=int, default=100)
parser.add_argument("--min-length", type=int, default=GENERATE_KWARGS["min_length"], help="min summary tokens (the worker's is 30, the model's default 56)")
parser.add_argument("--max-length", type=int, default=GENERATE_KWARGS["max_length"], help="max summary tokens (the worker's is 130, the model's default 142)")
parser.add_argument("--all-lengths", action="store_true",
                    help="use the first --num-docs test articles of any length, not only those the worker accepts "
                         "(closer to how published ROUGE scores are measured)")
parser.add_argument("--out", default=RESULTS_DIR / "beams_experiment.csv")
args = parser.parse_args()
generate_kwargs = {"min_length": args.min_length, "max_length": args.max_length}

documents = load_documents(args.num_docs, max_chars=None if args.all_lengths else MAX_DOCUMENT_CHARS)
device = resolve_device()
tokenizer, model = load_model(device)
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
writer = CsvWriter(args.out, environment())

means = {}
for num_beams in args.beams:
    # Warmup, so the first measured Document doesn't pay for kernel/cache setup at this beam count
    input = tokenizer(documents[0][0], return_tensors="pt", truncation=True).to(device)
    model.generate(**input, **generate_kwargs, num_beams=num_beams)

    rows = []
    for doc_index, (article, highlights) in enumerate(documents):
        input = tokenizer(article, return_tensors="pt", truncation=True).to(device)
        t1 = time.perf_counter()
        output = model.generate(**input, **generate_kwargs, num_beams=num_beams)
        if device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        summary = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        # rougeLsum splits on newlines: highlights already have one sentence per line, and the
        # model separates its sentences with " . "
        scores = scorer.score(highlights, summary.replace(" . ", " .\n"))
        row = {
            "num_beams": num_beams,
            "min_length": args.min_length,
            "max_length": args.max_length,
            "all_lengths": args.all_lengths,
            "doc_index": doc_index,
            "latency_s": t2 - t1,
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeLsum": scores["rougeLsum"].fmeasure,
            "summary": summary,
        }
        writer.write(row)
        rows.append(row)

    means[num_beams] = {key: sum(r[key] for r in rows) / len(rows) for key in ("latency_s", "rouge1", "rouge2", "rougeLsum")}
    print(f"num_beams={num_beams} done")

writer.close()
print(f"\n{len(documents)} Documents{' (any length)' if args.all_lengths else ''} on {device}, summaries of {args.min_length}-{args.max_length} tokens")
print(f"{'beams':>5}  {'latency_s':>9}  {'rouge1':>6}  {'rouge2':>6}  {'rougeLsum':>9}")
for num_beams, m in means.items():
    print(f"{num_beams:>5}  {m['latency_s']:>9.3f}  {m['rouge1']:>6.4f}  {m['rouge2']:>6.4f}  {m['rougeLsum']:>9.4f}")
