from src.cot_utils import *
from src.dataset_utils import *

# Bloomz model
model_name = "bigscience/bloomz-7b1-mt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

###### hf dataset ####### en,fr,es,zh
df = get_dataset_df("mgsm","en")

###### translated dataset ####### Portuguese, Catalan, Vietnamese, Indonesian
# dataset = get_translated_dataset_df("mgsm","Portuguese")

mgsm_generate_response(df=df,
                        task_lang="English",        # source language 
                        prompt_setting="basic",     # 'basic' or 'cot'
                        nr_shots=2,                 # 0-8 shots
                        shots_lang="English",       # select exemplars in this language (relevant when nr_shots > 0)
                        cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
                        model=model,                
                        tokenizer=tokenizer,
                        name="bloomz")              # model name for saving to .csv