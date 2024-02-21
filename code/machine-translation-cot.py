from src.translation_utils import *

model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name,token=True,src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,token=True)

llama_lang = ['bul_Cyrl','cat_Latn','ces_Latn','dan_Latn','eng_Latn','fin_Latn','hrv_Latn','hun_Latn','ind_Latn','ita_Latn','kor_Hang','nld_Latn','nno_Latn','pol_Latn','por_Latn','ron_Latn','slv_Latn','srp_Cyrl','swe_Latn','ukr_Cyrl','vie_Latn']

translate_cot_prompt(inputstring="Let's think step by step in ",
                     languages=llama_lang[:4],
                     model=model,
                     tokenizer=tokenizer)

