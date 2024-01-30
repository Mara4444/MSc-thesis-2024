from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, NllbTokenizer
import torch
from datasets import load_dataset
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

##### available languages #####
#   ace_Arab, ace_Latn, acm_Arab, acq_Arab, aeb_Arab, afr_Latn, ajp_Arab,
#   aka_Latn, amh_Ethi, apc_Arab, arb_Arab, ars_Arab, ary_Arab, arz_Arab,
#   asm_Beng, ast_Latn, awa_Deva, ayr_Latn, azb_Arab, azj_Latn, bak_Cyrl,
#   bam_Latn, ban_Latn,bel_Cyrl, bem_Latn, ben_Beng, bho_Deva, bjn_Arab, bjn_Latn,
#   bod_Tibt, bos_Latn, bug_Latn, bul_Cyrl, cat_Latn, ceb_Latn, ces_Latn,
#   cjk_Latn, ckb_Arab, crh_Latn, cym_Latn, dan_Latn, deu_Latn, dik_Latn,
#   dyu_Latn, dzo_Tibt, ell_Grek, eng_Latn, epo_Latn, est_Latn, eus_Latn,
#   ewe_Latn, fao_Latn, pes_Arab, fij_Latn, fin_Latn, fon_Latn, fra_Latn,
#   fur_Latn, fuv_Latn, gla_Latn, gle_Latn, glg_Latn, grn_Latn, guj_Gujr,
#   hat_Latn, hau_Latn, heb_Hebr, hin_Deva, hne_Deva, hrv_Latn, hun_Latn,
#   hye_Armn, ibo_Latn, ilo_Latn, ind_Latn, isl_Latn, ita_Latn, jav_Latn,
#   jpn_Jpan, kab_Latn, kac_Latn, kam_Latn, kan_Knda, kas_Arab, kas_Deva,
#   kat_Geor, knc_Arab, knc_Latn, kaz_Cyrl, kbp_Latn, kea_Latn, khm_Khmr,
#   kik_Latn, kin_Latn, kir_Cyrl, kmb_Latn, kon_Latn, kor_Hang, kmr_Latn,
#   lao_Laoo, lvs_Latn, lij_Latn, lim_Latn, lin_Latn, lit_Latn, lmo_Latn,
#   ltg_Latn, ltz_Latn, lua_Latn, lug_Latn, luo_Latn, lus_Latn, mag_Deva,
#   mai_Deva, mal_Mlym, mar_Deva, min_Latn, mkd_Cyrl, plt_Latn, mlt_Latn,
#   mni_Beng, khk_Cyrl, mos_Latn, mri_Latn, zsm_Latn, mya_Mymr, nld_Latn,
#   nno_Latn, nob_Latn, npi_Deva, nso_Latn, nus_Latn, nya_Latn, oci_Latn,
#   gaz_Latn, ory_Orya, pag_Latn, pan_Guru, pap_Latn, pol_Latn, por_Latn,
#   prs_Arab, pbt_Arab, quy_Latn, ron_Latn, run_Latn, rus_Cyrl, sag_Latn,
#   san_Deva, sat_Beng, scn_Latn, shn_Mymr, sin_Sinh, slk_Latn, slv_Latn,
#   smo_Latn, sna_Latn, snd_Arab, som_Latn, sot_Latn, spa_Latn, als_Latn,
#   srd_Latn, srp_Cyrl, ssw_Latn, sun_Latn, swe_Latn, swh_Latn, szl_Latn,
#   tam_Taml, tat_Cyrl, tel_Telu, tgk_Cyrl, tgl_Latn, tha_Thai, tir_Ethi,
#   taq_Latn, taq_Tfng, tpi_Latn, tsn_Latn, tso_Latn, tuk_Latn, tum_Latn,
#   tur_Latn, twi_Latn, tzm_Tfng, uig_Arab, ukr_Cyrl, umb_Latn, urd_Arab,
#   uzn_Latn, vec_Latn, vie_Latn, war_Latn, wol_Latn, xho_Latn, ydd_Hebr,
#   yor_Latn, yue_Hant, zho_Hans, zho_Hant, zul_Latn

def get_dataset(name,lang):
    """
    Loads a dataset from huggingface in the requested language.
    
    Parameters:
    name: name of the dataset ['mgsm', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum']
    lang: language of the dataset to load.
    
    Returns:
    Dataset in the specified language.
    """
    if name == "mgsm":
        dataset = load_dataset("juletxara/mgsm",lang) 
        
        return dataset
    
    elif name == "xcopa" and lang == "en":
        dataset = load_dataset("pkavumba/balanced-copa")
        
        return dataset
    
    elif name == "xcopa":
        dataset = load_dataset("xcopa",lang) 
        
        return dataset 
    
    elif name == "xstorycloze":
        dataset = load_dataset("juletxara/xstory_cloze",lang)   
        
        return dataset
    
    elif name == "mkqa":
        dataset = load_dataset("mkqa")
        
        if lang in dataset["train"]["queries"][0].keys():
            questionlist = [language[lang] for language in dataset["train"]["queries"]]
            answerlist = [language[lang] for language in dataset["train"]["answers"]]
            
            dataset = {
                "train": {
                    "queries": questionlist,
                    "answers": answerlist,
                }
            }

            return dataset
        
        else:
            print("Language not found. Specify one of the following languages: ['ar', 'da', 'de', 'en', 'es', 'fi', 'fr', 'he', 'hu', 'it','ja', 'km', 'ko', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sv', 'th', 'tr', 'vi', 'zh'")
    
    elif name == "pawsx":
        dataset = load_dataset("paws-x",lang)    

        return dataset
    
    elif name == "xnli":
        dataset = load_dataset("xnli",lang)  

        return dataset
    
    elif name == "xlsum":        
        dataset = load_dataset("csebuetnlp/xlsum",lang) # ['amharic', 'arabic', 'azerbaijani', 'bengali', 'burmese', 'chinese_simplified', 'chinese_traditional', 'english', 'french', 'gujarati', 'hausa', 'hindi', 'igbo', 'indonesian', 'japanese', 'kirundi', 'korean', 'kyrgyz', 'marathi', 'nepali', 'oromo', 'pashto', 'persian', 'pidgin', 'portuguese', 'punjabi', 'russian', 'scottish_gaelic', 'serbian_cyrillic', 'serbian_latin', 'sinhala', 'somali', 'spanish', 'swahili', 'tamil', 'telugu', 'thai', 'tigrinya', 'turkish', 'ukrainian', 'urdu', 'uzbek', 'vietnamese', 'welsh', 'yoruba']

        return dataset
    
    else:
        print("Dataset name is not correctly specified. Please input 'mgsm', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum'.")

def translate_list(input_list,src_lang,trg_lang):
    """
    Translate a list from the source language to the target language.
    
    Parameters:
    input_list: input list of strings to translate.
    src_lang: language of input string given in iso2-code.
    trg_lang: target language given in iso2-code.
    
    Returns:
    Translated list of strings.
    """
    # download model
    model_name = "facebook/nllb-200-3.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=True,src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name,token=True)

    translated_list = []

    for string in input_list:
        
        translated_string = ""
        
        sentences = sent_tokenize(string)

        for sentence in sentences:
            # print(sentence)
            inputs = tokenizer(
                sentence, 
                return_tensors="pt"
                )
        
            translated_tokens = model.generate(
                **inputs, 
                forced_bos_token_id=tokenizer.lang_code_to_id[trg_lang], 
                max_length=100 # set to longer than max length of a sentence in dataset?
                )
            
            translated_sentence = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            # print(translated_sentence)
            translated_string = translated_string + translated_sentence + ' '
        # print(translated_string)
        translated_list.append(translated_string)
        print(translated_list)
    print(translated_list)
    return translated_list

def translate_dataset(dataset,name,src_lang,trg_lang): 
    """
    Translate a dataset from the source language to the target language.
    
    Parameters:
    dataset: dataset to translate.
    name: name of the dataset ['mgsm', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum']
    src_lang: language of input string given in iso2-code.
    trg_lang: target language given in iso2-code.
    
    Returns:
    Translated dataset and returns as DataFrame. 
    """
    if name  == 'mgsm': 
        
        translated1_list = translate_list(dataset["test"]["question"],src_lang,trg_lang)

        translated_dataset = pd.DataFrame({'question': translated1_list,
                                           'answer_number': dataset["test"]["answer_number"]
                                           })

        translated_dataset.to_csv('./translations/mgsm_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset   
      
    elif name  == 'xcopa': 
        
        translated1_list = translate_list(dataset["test"]["premise"],src_lang,trg_lang)
        translated2_list = translate_list(dataset["test"]["question"],src_lang,trg_lang)
        translated3_list = translate_list(dataset["test"]["choice1"],src_lang,trg_lang)
        translated4_list = translate_list(dataset["test"]["choice2"],src_lang,trg_lang)

        translated_dataset = pd.DataFrame({'premise': translated1_list,
                                           'question': translated2_list,
                                           'choice1': translated3_list,
                                           'choice2': translated4_list, 
                                           'label': dataset["test"]["label"]
                                           })

        translated_dataset.to_csv('./translations/xcopa_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset
    
    elif name  == 'xstorycloze': 

        translated1_list = translate_list(dataset["eval"]["input_sentence_1"],src_lang,trg_lang)
        translated2_list = translate_list(dataset["eval"]["input_sentence_2"],src_lang,trg_lang)
        translated3_list = translate_list(dataset["eval"]["input_sentence_3"],src_lang,trg_lang)
        translated4_list = translate_list(dataset["eval"]["input_sentence_4"],src_lang,trg_lang)
        translated5_list = translate_list(dataset["eval"]["sentence_quiz1"],src_lang,trg_lang)
        translated6_list = translate_list(dataset["eval"]["sentence_quiz2"],src_lang,trg_lang)
        
        translated_dataset = pd.DataFrame({'input_sentence_1': translated1_list,
                                           'input_sentence_2': translated2_list,
                                           'input_sentence_3': translated3_list,
                                           'input_sentence_4': translated4_list, 
                                           'sentence_quiz1': translated5_list,
                                           'sentence_quiz1': translated6_list,
                                           'answer_right_ending': dataset["eval"]["answer_right_ending"]
                                           })

        translated_dataset.to_csv('./translations/xstorycloze_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset
    
    # elif name == "mkqa":
    # answer column is in this shape: [{'type': 5, 'entity': '', 'text': '11.0 years', 'aliases': ['11 years']}]
    # how to translate only text and aliases and keep the rest of the structure?

    elif name  == 'pawsx': 
        
        translated1_list = translate_list(dataset["test"]["sentence1"],src_lang,trg_lang)
        translated2_list = translate_list(dataset["test"]["sentence2"],src_lang,trg_lang)

        translated_dataset = pd.DataFrame({'sentence1': translated1_list,
                                           'sentence2': translated2_list,
                                           'label': dataset["test"]["label"]
                                           })

        translated_dataset.to_csv('./translations/pawsx_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset
    
    elif name  == 'xnli': 
        
        translated1_list = translate_list(dataset["test"]["premise"],src_lang,trg_lang)
        translated2_list = translate_list(dataset["test"]["hypothesis"],src_lang,trg_lang)

        translated_dataset = pd.DataFrame({'premise': translated1_list,
                                           'hypothesis': translated2_list,
                                           'label': dataset["test"]["label"]
                                           })

        translated_dataset.to_csv('./translations/xnli_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset
    
    elif name  == 'xlsum': 
        
        translated1_list = translate_list(dataset["test"]["title"],src_lang,trg_lang)
        translated2_list = translate_list(dataset["test"]["summary"],src_lang,trg_lang)
        translated3_list = translate_list(dataset["test"]["text"],src_lang,trg_lang)

        translated_dataset = pd.DataFrame({'title': translated1_list,
                                           'summary': translated2_list,
                                           'text': translated3_list
                                           })

        translated_dataset.to_csv('./translations/xlsum_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset

    else:
        print("Dataset name is not correctly specified. Please input 'mgsm', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum'.")

translate_dataset(get_dataset("mgsm","en"),"mgsm","eng_Latn","nld_Latn")
   