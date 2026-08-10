from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
from datasets import load_dataset


# You can change this to any other model you want to use
# For news summarization, I recommend using "sshleifer/distilbart-cnn-12-6" (CPU) or "facebook/bart-large-cnn"
# For research paper/ticket summarization, use something else
model_name = "sshleifer/distilbart-cnn-12-6"

# Pulling 20 samples from the CNN/DailyMail dataset for testing
articles = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test[:20]")

# number of beams to test (how deeply to search for the best summary)
num_beams_arr = [2, 4] # Can always add more

# Timing the model loading time
t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
t1 = time.perf_counter()
print(f"Model loaded in: {t1 - t0:.2f} seconds")


for num_beams in num_beams_arr:
    print(f"\n\nRunning inference with num_beams={num_beams}...\n")
    # Warmup call to load the model into memory
    # Has to be called in this loop, because the model has a new cold start due to change in num_beams
    article = articles[0]["article"]
    input = tokenizer(article, return_tensors="pt", truncation=True)
    model.generate(**input, max_length=130, min_length=30, num_beams=num_beams)

    # Loop through the articles and summarize each one (with timing)
    for row in articles:
        text = row["article"]
        input = tokenizer(text, return_tensors="pt", truncation=True)
        t2 = time.perf_counter()
        summary = model.generate(**input, max_length=130, min_length=30, num_beams=num_beams)
        t3 = time.perf_counter()
        print(f"Inference time: {t3 - t2:.2f} seconds")
        print("Summary:", tokenizer.decode(summary[0], skip_special_tokens=True))
        print("Highlights (human):", row["highlights"])
