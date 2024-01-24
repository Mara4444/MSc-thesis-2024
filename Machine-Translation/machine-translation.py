from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# download model
model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# Specify source and target languages
src_lang = 'en'  # Source language (e.g., English)
tgt_lang = 'fr'  # Target language (e.g., French)

input_text = "Translate this sentence."
input_ids = tokenizer.encode(input_text, return_tensors="pt", languages=src_lang)

# Perform inference
output_ids = model.generate(input_ids, target_language=tgt_lang)

# Decode output
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Translated from {src_lang} to {tgt_lang}:", output_text)

