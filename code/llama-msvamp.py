from src.cot_utils import *
from src.dataset_utils import *

# Llama-2 model

# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# ##### hf dataset ####### en,fr,es,zh ----- te,th,de,sw,bn,ru,ja
langs = ['fr','es','zh','te','th','de','sw','bn','ru','ja']

for lang in langs:

      dataset = get_dataset_df("msvamp",lang)

      generate_response(df=dataset,
                        task='msvamp',
                              task_lang=lang,        # source language 
                              instr_lang="English",       # get instruction prompt in this language
                              prompt_setting="basic",     # 'basic' or 'cot'
                              model=model,                
                              tokenizer=tokenizer,
                              name="llama-2-13b")           # model name for saving to .csv

      generate_response(df=dataset,
                        task='msvamp',
                              task_lang=lang,        # source language 
                              instr_lang="English",       # get instruction prompt in this language
                              prompt_setting="cot",     # 'basic' or 'cot'
                              model=model,                
                              tokenizer=tokenizer,
                              name="llama-2-13b")           # model name for saving to .csv
