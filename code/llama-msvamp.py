from src.cot_utils import *
from src.dataset_utils import *

# Bloomz model

model_name = "bigscience/bloomz-7b1-mt"
model = BloomForCausalLM.from_pretrained(model_name)
tokenizer = BloomTokenizerFast.from_pretrained(model_name)

###### hierna alle langs met llama basic prompt english instruct testen

# ##### hf dataset ####### en,fr,es,zh ----- te,th,de,sw,bn,ru,ja
English = get_dataset_df("msvamp","en")

msvamp_generate_response(df=English,
                        task_lang="English",        # source language 
                        instr_lang="English",
                        prompt_setting="basic",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="bloomz-7b1")              # model name for saving to .csv

msvamp_generate_response(df=English,
                        task_lang="English",        # source language 
                        instr_lang="English",
                        prompt_setting="cot",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="bloomz-7b1")              # model name for saving to .csv

# Llama-2 model

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

msvamp_generate_response(df=English,
                        task_lang="English",        # source language 
                        instr_lang="English",
                        prompt_setting="basic",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="llama2-7b")              # model name for saving to .csv

msvamp_generate_response(df=English,
                        task_lang="English",        # source language 
                        instr_lang="English",
                        prompt_setting="cot",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="llama2-7b")              # model name for saving to .csv
