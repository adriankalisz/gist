from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
from app.config import settings

tokenizer = None
model = None

# Loads the model and tokenizer into memory
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.model_name)
    # Warmup call to load the model into memory
    input = tokenizer("""This is just a call to load the model""", return_tensors="pt", truncation=True)
    model.generate(**input, max_length=130, min_length=30, num_beams=settings.num_beams)

# Summarizes the message, and times the inference time. Returns a tuple of (summary, inference_time)
def summarize(text: str, num_beams: int = settings.num_beams) -> tuple[str, float]:
    input = tokenizer(text, return_tensors="pt", truncation=True)
    t2 = time.perf_counter()
    summary = model.generate(**input, max_length=130, min_length=30, num_beams=num_beams)
    t3 = time.perf_counter()
    return tokenizer.decode(summary[0], skip_special_tokens=True), t3 - t2