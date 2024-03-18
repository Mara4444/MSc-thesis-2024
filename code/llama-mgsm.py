from src.cot_utils import *
from src.dataset_utils import *

# Llama-2 model

model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# mgsm_langs = {'en' : 'English',
#                 'fr' : 'French',
#                 'es' : 'Spanish',
#                 'te' : 'Telugu',
#                 'de' : 'German',
#                 'bn' : 'Bengali',
#                 'sw' : 'Swahili',
#                 'ru' : 'Russian',
#                 'th' : 'Thai',
#                 'zh' : 'Chinese',
#                 'ja' : 'Japanese'}


# mt_langs = ['Afrikaans','Arabic','Balinese','Belarusian','Tibetan', 'Bosnian', 'Haitian','Indonesian','Quechua',
#                                  'Bulgarian', 'Catalan', 'Czech', 'Danish', 'Estonian','Khmer', 'Korean', 'Lao', 'Maithili', 
#                                  'Malayalam', 'Marathi', 'Dutch', 'Norwegian', 'Nepali', 'Polish', 'Greek',
#                                  'Portuguese','Romanian','Finnish','Hebrew','Slovak','Hindi','Vietnamese',
#                                  'Croatian','Hungarian','Swedish','Javanese',"Armenian", "Bulgarian", 'Turkish',
#                                  "Burmese", "Cantonese", "Malay", "Serbian", "Slovenian", "Tagalog", 'Tamil',
#                                  "Ukrainian", "Urdu", "Zulu"]

# mt_langs = ['Catalan', 'Czech', 'Danish', 'Estonian','Khmer', 'Korean', 'Lao', 'Maithili', 
#                                  'Malayalam', 'Marathi', 'Dutch', 'Norwegian', 'Nepali', 'Polish', 'Greek',
#                                  'Portuguese','Romanian','Finnish','Hebrew','Slovak','Hindi','Vietnamese']

mt_langs = ['Croatian','Hungarian','Swedish','Javanese',"Armenian", "Bulgarian", 'Turkish',
                                 "Burmese", "Cantonese", "Malay", "Serbian", "Slovenian", "Tagalog", 'Tamil',
                                 "Ukrainian", "Urdu", "Zulu"]

for lang in mt_langs:

      dataset = get_translated_dataset_df("mgsm",lang)

      generate_response(df=dataset,
                        task='mgsm',
                              task_lang=lang,        # source language 
                              instr_lang="English",       # get instruction prompt in this language
                              prompt_setting="basic",     # 'basic' or 'cot'
                              model=model,                
                              tokenizer=tokenizer,
                              name="llama-2-7b")           # model name for saving to .csv

      generate_response(df=dataset,
                        task='mgsm',
                              task_lang=lang,        # source language 
                              instr_lang="English",       # get instruction prompt in this language
                              prompt_setting="cot",     # 'basic' or 'cot'
                              model=model,                
                              tokenizer=tokenizer,
                              name="llama-2-7b")           # model name for saving to .csv
      
