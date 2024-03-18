from src.translation_utils import *

model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name,token=True,src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,token=True)

langs = ["afr_Latn","arb_Arab","ban_Latn","bel_Cyrl","ben_Beng","bod_Tibt", "bos_Latn","bul_Cyrl",
"ces_Latn", "cat_Latn","dan_Latn", "deu_Latn","eng_Latn","ell_Grek","est_Latn", 
"fin_Latn", "fra_Latn","hat_Latn", "heb_Hebr","hin_Deva","hun_Latn", "hrv_Latn", "hye_Armn", 
"ind_Latn", "ita_Latn","jav_Latn", "jpn_Jpan","khm_Khmr","kor_Hang", 
"lao_Laoo","mai_Deva", "mal_Mlym", "mar_Deva", "mya_Mymr", "nno_Latn",
"nld_Latn", "npi_Deva","pol_Latn","por_Latn", "slk_Latn","quy_Latn","ron_Latn", "rus_Cyrl", 
"slv_Latn", "spa_Latn", "srp_Cyrl", "swe_Latn", "swh_Latn", "tam_Taml", "tel_Telu", 
"tgl_Latn", 'tha_Thai',"tur_Latn","ukr_Cyrl", "urd_Arab", "vie_Latn" , 'yue_Hant', "zho_Hant", "zsm_Latn","zul_Latn"]

# translate_instruction_xcopa(languages=langs,
#                             model=model,
#                             tokenizer=tokenizer)

# translate_instruction_cot(languages=langs,
#                           model=model,
#                           tokenizer=tokenizer)

translate_instruction_mgsm(languages=langs,
                          model=model,
                          tokenizer=tokenizer)

# translate_instruction_coinflip(languages=langs,
#                           model=model,
#                           tokenizer=tokenizer)

