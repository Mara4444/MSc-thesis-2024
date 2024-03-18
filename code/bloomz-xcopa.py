from src.cot_utils import *
from src.dataset_utils import *

# Bloomz model

model_name = "bigscience/bloomz-7b1-mt"
model = BloomForCausalLM.from_pretrained(model_name)
tokenizer = BloomTokenizerFast.from_pretrained(model_name)

xcopa_langs = {'en' : 'English',
                'et' : 'Estonian',
                'ht' : 'Haitian',
                'id' : 'Indonesian',
                'qu' : 'Quechua',
                'it' : 'Italian',
                'sw' : 'Swahili',
                'ta' : 'Tamil',
                'th' : 'Thai',
                'zh' : 'Chinese',
                'tr' : 'Turkish',
                'vi' : 'Vietnamese'}

mt_langs = ['Afrikaans','Arabic','Balinese','Belarusian','Bengali','Tibetan', 'Bosnian', 
                                 'Bulgarian', 'Catalan', 'Czech', 'Danish', 'Khmer', 'Korean', 'Lao', 'Maithili', 
                                 'Malayalam', 'Marathi', 'Dutch', 'Norwegian', 'Nepali', 'German', 'Polish', 'Greek',
                                 'Portuguese','Russian','French','Romanian','Finnish','Hebrew','Slovak','Hindi',
                                 'Croatian','Hungarian','Swedish','Japanese','Javanese',"Armenian", "Bulgarian", 
                                 "Burmese", "Cantonese", "Malay", "Serbian", "Slovenian", "Spanish", "Tagalog", 
                                 "Telugu", "Ukrainian", "Urdu", "Zulu"]

# English = get_dataset_df("xcopa","en")

# for lang in xcopa_langs:
    
#     # generate_response(df = get_dataset_df("xcopa",lang),
#     #               task = "xcopa",
#     #               task_lang = xcopa_langs[lang],
#     #               instr_lang = xcopa_langs[lang],
#     #               prompt_setting = "basic",
#     #               model = model,
#     #               tokenizer = tokenizer,
#     #               name = "bloomz-7b1")
    
#     generate_response(df = English,
#                   task = "xcopa",
#                   task_lang = "English",
#                   instr_lang = xcopa_langs[lang],
#                   prompt_setting = "basic",
#                   model = model,
#                   tokenizer = tokenizer,
#                   name = "bloomz-7b1")
    
      
# for lang in mt_langs:

#     generate_response(df = English,
#                   task = "xcopa",
#                   task_lang = "English",
#                   instr_lang = lang,
#                   prompt_setting = "basic",
#                   model = model,
#                   tokenizer = tokenizer,
#                   name = "bloomz-7b1")

for lang in mt_langs:

    generate_response(df = get_translated_dataset_df("xcopa",lang),
                  task = "xcopa",
                  task_lang = lang,
                  instr_lang = lang,
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "bloomz-7b1")

