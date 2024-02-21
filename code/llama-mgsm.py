from src.cot_utils import *
from src.dataset_utils import *

# Llama-2 model

model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "meta-llama/Llama-2-13b-chat-hf"
# model_name = "meta-llama/Llama-2-70b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

###### translated dataset ####### Portuguese, Catalan, Vietnamese, Indonesian
# dataset = get_translated_dataset_df("mgsm","Portuguese")

###### hf dataset ####### en,fr,es,zh ----- te,th,de,sw,bn,ru,ja
# English = get_dataset_df("mgsm","en")

# mgsm_generate_response(df=English,
#                         task_lang="English",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=8,                 # 0-8 shots
#                         shots_lang="English",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# French = get_dataset_df("mgsm","fr")

# mgsm_generate_response(df=French,
#                         task_lang="French",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="French",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Spanish = get_dataset_df("mgsm","es")

# mgsm_generate_response(df=Spanish,
#                         task_lang="Spanish",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Spanish",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Chinese = get_dataset_df("mgsm","zh")

# mgsm_generate_response(df=Chinese,
#                         task_lang="Chinese",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Chinese",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# German = get_dataset_df("mgsm","de")

# mgsm_generate_response(df=German,
#                         task_lang="German",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="German",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Russian = get_dataset_df("mgsm","ru")

# mgsm_generate_response(df=Russian,
#                         task_lang="Russian",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Russian",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Japanese = get_dataset_df("mgsm","ja")

# mgsm_generate_response(df=Japanese,
#                         task_lang="Japanese",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Japanese",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Thai = get_dataset_df("mgsm","th")

# mgsm_generate_response(df=Thai,
#                         task_lang="Thai",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Thai",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Bengali = get_dataset_df("mgsm","bn")

# mgsm_generate_response(df=Bengali,
#                         task_lang="Bengali",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Bengali",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Telugu = get_dataset_df("mgsm","te")

# mgsm_generate_response(df=Telugu,
#                         task_lang="Telugu",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Telugu",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Swahili = get_dataset_df("mgsm","sw")

# mgsm_generate_response(df=Swahili,
#                         task_lang="Swahili",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Swahili",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv


# Portuguese = get_translated_dataset_df("mgsm","Portuguese")

# mgsm_generate_response(df=Portuguese,
#                         task_lang="Portuguese",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Portuguese",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Catalan = get_translated_dataset_df("mgsm","Catalan")

# mgsm_generate_response(df=Catalan,
#                         task_lang="Catalan",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Catalan",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Vietnamese = get_translated_dataset_df("mgsm","Vietnamese")

# mgsm_generate_response(df=Vietnamese,
#                         task_lang="Vietnamese",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Vietnamese",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Indonesian = get_translated_dataset_df("mgsm","Indonesian")

# mgsm_generate_response(df=Indonesian,
#                         task_lang="Indonesian",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Indonesian",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Bulgarian = get_translated_dataset_df("mgsm","Bulgarian")

# mgsm_generate_response(df=Bulgarian,
#                         task_lang="Bulgarian",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Bulgarian",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Czech = get_translated_dataset_df("mgsm","Czech")

# mgsm_generate_response(df=Czech,
#                         task_lang="Czech",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Czech",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Danish = get_translated_dataset_df("mgsm","Danish")

# mgsm_generate_response(df=Danish,
#                         task_lang="Danish",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Danish",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Finnish = get_translated_dataset_df("mgsm","Finnish")

# mgsm_generate_response(df=Finnish,
#                         task_lang="Finnish",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Finnish",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Croatian = get_translated_dataset_df("mgsm","Croatian")

# mgsm_generate_response(df=Croatian,
#                         task_lang="Croatian",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Croatian",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Hungarian = get_translated_dataset_df("mgsm","Hungarian")

# mgsm_generate_response(df=Hungarian,
#                         task_lang="Hungarian",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Hungarian",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Italian = get_translated_dataset_df("mgsm","Italian")

# mgsm_generate_response(df=Italian,
#                         task_lang="Italian",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Italian",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Korean = get_translated_dataset_df("mgsm","Korean")

# mgsm_generate_response(df=Korean,
#                         task_lang="Korean",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Korean",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

# Dutch = get_translated_dataset_df("mgsm","Dutch")

# mgsm_generate_response(df=Dutch,
#                         task_lang="Dutch",        # source language 
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         nr_shots=6,                 # 0-8 shots
#                         shots_lang="Dutch",       # select exemplars in this language (relevant when nr_shots > 0)
#                         cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama2-7b")              # model name for saving to .csv

Norwegian = get_translated_dataset_df("mgsm","Norwegian")

mgsm_generate_response(df=Norwegian,
                        task_lang="Norwegian",        # source language 
                        prompt_setting="cot",     # 'basic' or 'cot'
                        nr_shots=0,                 # 0-8 shots
                        shots_lang="Norwegian",       # select exemplars in this language (relevant when nr_shots > 0)
                        cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
                        model=model,                
                        tokenizer=tokenizer,
                        name="llama2-7b")              # model name for saving to .csv

Polish = get_translated_dataset_df("mgsm","Polish")

mgsm_generate_response(df=Polish,
                        task_lang="Polish",        # source language 
                        prompt_setting="cot",     # 'basic' or 'cot'
                        nr_shots=0,                 # 0-8 shots
                        shots_lang="Polish",       # select exemplars in this language (relevant when nr_shots > 0)
                        cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
                        model=model,                
                        tokenizer=tokenizer,
                        name="llama2-7b")              # model name for saving to .csv

Romanian = get_translated_dataset_df("mgsm","Romanian")

mgsm_generate_response(df=Romanian,
                        task_lang="Romanian",        # source language 
                        prompt_setting="cot",     # 'basic' or 'cot'
                        nr_shots=0,                 # 0-8 shots
                        shots_lang="Romanian",       # select exemplars in this language (relevant when nr_shots > 0)
                        cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
                        model=model,                
                        tokenizer=tokenizer,
                        name="llama2-7b")              # model name for saving to .csv

Slovenian = get_translated_dataset_df("mgsm","Slovenian")

mgsm_generate_response(df=Slovenian,
                        task_lang="Slovenian",        # source language 
                        prompt_setting="cot",     # 'basic' or 'cot'
                        nr_shots=0,                 # 0-8 shots
                        shots_lang="Slovenian",       # select exemplars in this language (relevant when nr_shots > 0)
                        cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
                        model=model,                
                        tokenizer=tokenizer,
                        name="llama2-7b")              # model name for saving to .csv

Serbian = get_translated_dataset_df("mgsm","Serbian")

mgsm_generate_response(df=Serbian,
                        task_lang="Serbian",        # source language 
                        prompt_setting="cot",     # 'basic' or 'cot'
                        nr_shots=0,                 # 0-8 shots
                        shots_lang="Serbian",       # select exemplars in this language (relevant when nr_shots > 0)
                        cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
                        model=model,                
                        tokenizer=tokenizer,
                        name="llama2-7b")              # model name for saving to .csv

Swedish = get_translated_dataset_df("mgsm","Swedish")

mgsm_generate_response(df=Swedish,
                        task_lang="Swedish",        # source language 
                        prompt_setting="cot",     # 'basic' or 'cot'
                        nr_shots=0,                 # 0-8 shots
                        shots_lang="Swedish",       # select exemplars in this language (relevant when nr_shots > 0)
                        cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
                        model=model,                
                        tokenizer=tokenizer,
                        name="llama2-7b")              # model name for saving to .csv

Ukrainian = get_translated_dataset_df("mgsm","Ukrainian")

mgsm_generate_response(df=Ukrainian,
                        task_lang="Ukrainian",        # source language 
                        prompt_setting="cot",     # 'basic' or 'cot'
                        nr_shots=0,                 # 0-8 shots
                        shots_lang="Ukrainian",       # select exemplars in this language (relevant when nr_shots > 0)
                        cot_lang="",                # reasoning language (relevant when prompsetting = 'cot')
                        model=model,                
                        tokenizer=tokenizer,
                        name="llama2-7b")              # model name for saving to .csv
