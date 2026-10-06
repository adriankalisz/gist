from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import torch
from app.config import settings

tokenizer = None
model = None
device = None

def is_ready() -> bool:
    return tokenizer is not None and model is not None

# Resolves the "auto" device setting to cuda when available, otherwise cpu
def resolve_device() -> str:
    if settings.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return settings.device

# Loads the model and tokenizer into memory
def load_model():
    global tokenizer, model, device
    device = resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.model_name).to(device)
    # Warmup call to load the model into memory
    input = tokenizer("""This is just a call to load the model""", return_tensors="pt", truncation=True).to(device)
    model.generate(**input, max_length=142, min_length=56, num_beams=settings.num_beams)


# Summarizes the message, and times the inference time. Returns a tuple of (summaries (List[str]), inference_time)
def summarize(articles: list[str]) -> tuple[list[str], float]:
    input = tokenizer(articles, return_tensors="pt", truncation=True, padding=True).to(device)
    t2 = time.perf_counter()
    summaries = model.generate(**input, max_length=130, min_length=30, num_beams=settings.num_beams)
    t3 = time.perf_counter()
    return [tokenizer.decode(summary, skip_special_tokens=True) for summary in summaries], t3 - t2

