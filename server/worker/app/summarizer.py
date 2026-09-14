from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
from app.config import settings

tokenizer = None
model = None

def is_ready() -> bool:
    return tokenizer is not None and model is not None

# Loads the model and tokenizer into memory
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.model_name)
    # Warmup call to load the model into memory
    input = tokenizer("""This is just a call to load the model""", return_tensors="pt", truncation=True)
    model.generate(**input, max_length=130, min_length=30, num_beams=settings.num_beams)


# Summarizes the message, and times the inference time. Returns a tuple of (summaries (List[str]), inference_time)
def summarize(articles: list[str], num_beams: int | None = None) -> tuple[list[str], float]:
    if num_beams is None:
        num_beams = settings.num_beams
    input = tokenizer(articles, return_tensors="pt", truncation=True, padding=True)
    t2 = time.perf_counter()
    summaries = model.generate(**input, max_length=130, min_length=30, num_beams=num_beams)
    t3 = time.perf_counter()
    return [tokenizer.decode(summary, skip_special_tokens=True) for summary in summaries], t3 - t2
    
