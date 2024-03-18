from src.cot_utils import *
from src.dataset_utils import *

# Llama-2 model

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# xcopa_langs = {'en' : 'English',
#                 'et' : 'Estonian',
#                 'ht' : 'Haitian',
#                 'id' : 'Indonesian',
#                 'qu' : 'Quechua',
#                 'it' : 'Italian',
#                 'sw' : 'Swahili',
#                 'ta' : 'Tamil',
#                 'th' : 'Thai',
#                 'zh' : 'Chinese',
#                 'tr' : 'Turkish',
#                 'vi' : 'Vietnamese'}

# xcopa_langs = {'et' : 'Estonian',
#                 'ht' : 'Haitian',
#                 'id' : 'Indonesian',
#                 'qu' : 'Quechua',
#                 'sw' : 'Swahili'}

# xcopa_langs = { 'ta' : 'Tamil',
#                 'th' : 'Thai',
#                 'it' : 'Italian'}

# xcopa_langs = { 'zh' : 'Chinese',
#                 'tr' : 'Turkish',
#                 'vi' : 'Vietnamese'}


# mt_langs = ['Afrikaans','Arabic','Balinese','Belarusian','Bengali','Tibetan', 'Bosnian', 
#                                  'Bulgarian', 'Catalan', 'Czech', 'Danish', 'Khmer', 'Korean', 'Lao', 'Maithili', 
#                                  'Malayalam', 'Marathi', 'Dutch', 'Norwegian', 'Nepali', 'German', 'Polish', 'Greek',
#                                  'Portuguese','Russian','French','Romanian','Finnish','Hebrew','Slovak','Hindi',
#                                  'Croatian','Hungarian','Swedish','Japanese','Javanese',"Armenian", "Bulgarian", 
#                                  "Burmese", "Cantonese", "Malay", "Serbian", "Slovenian", "Spanish", "Tagalog", 
#                                  "Telugu", "Ukrainian", "Urdu", "Zulu"]

# mt_langs = ['Malayalam', 'German', 'French','Hindi','Swedish','Javanese', "Burmese", "Cantonese","Telugu"]
mt_langs = ["German","Telugu"]

for lang in mt_langs:

    generate_response(df = get_translated_dataset_df("xcopa",lang),
                  task = "xcopa",
                  task_lang = lang,
                  instr_lang = lang,
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "llama-2-7b")
    
# for lang in xcopa_langs:
    
#     generate_response(df = get_dataset_df("xcopa",lang),
#                   task = "xcopa",
#                   task_lang = xcopa_langs[lang],
#                   instr_lang = xcopa_langs[lang],
#                   prompt_setting = "basic",
#                   model = model,
#                   tokenizer = tokenizer,
#                   name = "llama-2-7b")

# mt_langs = ["Burmese", "Cantonese", "Malay", "Serbian", "Slovenian", "Spanish", "Tagalog", 
#                                  "Telugu", "Ukrainian", "Urdu", "Zulu"]
    
# English = get_dataset_df("xcopa","en")

# for lang in mt_langs:
    
#     generate_response(df = English,
#                   task = "xcopa",
#                   task_lang = "English",
#                   instr_lang = lang,
#                   prompt_setting = "basic",
#                   model = model,
#                   tokenizer = tokenizer,
#                   name = "llama-2-7b")
    
  
