from src.cot_utils import *
from src.dataset_utils import *

# Llama-2 model

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

###### translated dataset ####### Portuguese, Catalan, Vietnamese, Indonesian
# dataset = get_translated_dataset_df("mgsm","Portuguese")

##### hf dataset ####### en,fr,es,zh ----- te,th,de,sw,bn,ru,ja
English = get_dataset_df("mgsm","en")

mgsm_generate_response(df=English,
                        task_lang="English",        # source language 
                        instr_lang="English",       # get instruction prompt in this language
                        prompt_setting="basic",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="llama2-7b")           # model name for saving to .csv

mgsm_generate_response(df=English,
                        task_lang="English",        # source language 
                        instr_lang="English",       # get instruction prompt in this language
                        prompt_setting="cot",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="llama2-7b")           # model name for saving to .csv

