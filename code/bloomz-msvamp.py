from src.cot_utils import *
from src.dataset_utils import *

# Bloomz model

model_name = "bigscience/bloomz-7b1-mt"
model = BloomForCausalLM.from_pretrained(model_name)
tokenizer = BloomTokenizerFast.from_pretrained(model_name)

# ##### hf dataset ####### en,fr,es,zh ----- te,th,de,sw,bn,ru,ja
# English = get_dataset_df("msvamp","en")

# msvamp_generate_response(df=English,
#                         task_lang="English",        # source language 
#                         instr_lang="English",
#                         prompt_setting="basic",     # 'basic' or 'cot'
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="bloomz-7b1")              # model name for saving to .csv

French = get_dataset_df("msvamp","fr")

msvamp_generate_response(df=French,
                        task_lang="French",        # source language 
                        instr_lang="English",
                        prompt_setting="basic",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="bloomz-7b1")              # model name for saving to .csv

Spanish = get_dataset_df("msvamp","es")

msvamp_generate_response(df=Spanish,
                        task_lang="Spanish",        # source language 
                        instr_lang="English",
                        prompt_setting="basic",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="bloomz-7b1")              # model name for saving to .csv

Chinese = get_dataset_df("msvamp","zh")

msvamp_generate_response(df=Chinese,
                        task_lang="Chinese",        # source language 
                        instr_lang="English",
                        prompt_setting="basic",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="bloomz-7b1")              # model name for saving to .csv

Telugu = get_dataset_df("msvamp","te")

msvamp_generate_response(df=Telugu,
                        task_lang="Telugu",        # source language 
                        instr_lang="English",
                        prompt_setting="basic",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="bloomz-7b1")              # model name for saving to .csv

Thai = get_dataset_df("msvamp","th")

msvamp_generate_response(df=Thai,
                        task_lang="Thai",        # source language 
                        instr_lang="English",
                        prompt_setting="basic",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="bloomz-7b1")              # model name for saving to .csv

German = get_dataset_df("msvamp","de")

msvamp_generate_response(df=German,
                        task_lang="German",        # source language 
                        instr_lang="English",
                        prompt_setting="basic",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="bloomz-7b1")              # model name for saving to .csv

Swahili = get_dataset_df("msvamp","sw")

msvamp_generate_response(df=Swahili,
                        task_lang="English",        # source language 
                        instr_lang="English",
                        prompt_setting="basic",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="bloomz-7b1")              # model name for saving to .csv

Bengali = get_dataset_df("msvamp","be")

msvamp_generate_response(df=Bengali,
                        task_lang="Bengali",        # source language 
                        instr_lang="English",
                        prompt_setting="basic",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="bloomz-7b1")              # model name for saving to .csv

Russian = get_dataset_df("msvamp","ru")

msvamp_generate_response(df=Russian,
                        task_lang="Russian",        # source language 
                        instr_lang="English",
                        prompt_setting="basic",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="bloomz-7b1")              # model name for saving to .csv

Japanese = get_dataset_df("msvamp","ja")

msvamp_generate_response(df=Japanese,
                        task_lang="Japanese",        # source language 
                        instr_lang="English",
                        prompt_setting="basic",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="bloomz-7b1")              # model name for saving to .csv
