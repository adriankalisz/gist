from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

model_name = "sshleifer/distilbart-cnn-12-6"  # You can change this to any other model you want to use

t0 = time.perf_counter()
# Initialize the summarization pipeline with the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
t1 = time.perf_counter()
print(f"Model loaded in: {t1 - t0:.2f} seconds")

article = """(CNN) -- An American woman died aboard a cruise ship that docked at Rio de Janeiro on Tuesday, the same ship on which 86 passengers previously fell ill, according to the state-run Brazilian news agency, Agencia Brasil. The American tourist died aboard the MS Veendam, owned by cruise operator Holland America. Federal Police told Agencia Brasil that forensic doctors were investigating her death. The ship's doctors told police that the woman was elderly and suffered from diabetes and hypertension, according the agency. The other passengers came down with diarrhea prior to her death during an earlier part of the trip, the ship's doctors said. The Veendam left New York 36 days ago for a South America tour."""

input = tokenizer(article, return_tensors="pt", truncation=True)
model.generate(**input, max_length=130, min_length=30)  # Warm-up run to load the model into memory
t2 = time.perf_counter()
summary = model.generate(**input, max_length=130, min_length=30)
t3 = time.perf_counter()
print(f"Inference time: {t3 - t2:.2f} seconds")
print("Summary:", tokenizer.decode(summary[0], skip_special_tokens=True))
