
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datasets import load_dataset
import pandas as pd

language_codes = {'Acehnese (Arabic script)' :	'ace_Arab',
    'Acehnese' : 'ace_Latn',
    'Mesopotamian': 'acm_Arab',
    "Ta'izzi-Adeni": 'acq_Arab',
    'Tunisian': 'aeb_Arab',
    'Afrikaans': 'afr_Latn',
    'South Levantine': 'ajp_Arab',
    'Akan': 'aka_Latn',
    'Amharic': 'amh_Ethi',
    'North Levantine': 'apc_Arab',
    'Arabic': 'arb_Arab',
    'Arabic (Romanized)': 'arb_Latn',
    'Najdi': 'ars_Arab',
    'Moroccan': 'ary_Arab',
    'Egyptian': 'arz_Arab',
    'Assamese': 'asm_Beng',
    'Asturian': 'ast_Latn',
    'Awadhi': 'awa_Deva',
    'Aymara': 'ayr_Latn',
    'South Azerbaijani': 'azb_Arab',
    'North Azerbaijani': 'azj_Latn',
    'Bashkir': 'bak_Cyrl',
    'Bambara': 'bam_Latn',
    'Balinese': 'ban_Latn',
    'Belarusian': 'bel_Cyrl',
    'Bemba': 'bem_Latn',
    'Bengali': 'ben_Beng',
    'Bhojpuri': 'bho_Deva',
    'Banjar (Arabic script)': 'bjn_Arab',
    'Banjar': 'bjn_Latn',
    'Tibetan': 'bod_Tibt',
    'Bosnian': 'bos_Latn',
    'Buginese': 'bug_Latn',
    'Bulgarian': 'bul_Cyrl',
    'Catalan': 'cat_Latn',
    'Cebuano': 'ceb_Latn',
    'Czech': 'ces_Latn',
    'Chokwe': 'cjk_Latn',
    'Central Kurdish': 'ckb_Arab',
    'Tatar': 'crh_Latn',
    'Welsh': 'cym_Latn',
    'Danish': 'dan_Latn',
    'German': 'deu_Latn',
    'Dinka': 'dik_Latn',
    'Dyula': 'dyu_Latn',
    'Dzongkha': 'dzo_Tibt',
    'Greek': 'ell_Grek',
    'English': 'eng_Latn',
    'Esperanto': 'epo_Latn',
    'Estonian': 'est_Latn',
    'Basque': 'eus_Latn',
    'Ewe': 'ewe_Latn',
    'Faroese': 'fao_Latn',
    'Fijian': 'fij_Latn',
    'Finnish': 'fin_Latn',
    'Fon': 'fon_Latn',
    'French': 'fra_Latn',
    'Friulian': 'fur_Latn',
    'Fulfulde': 'fuv_Latn',
    'Gaelic': 'gla_Latn',
    'Irish': 'gle_Latn',
    'Galician': 'glg_Latn',
    'Guarani': 'grn_Latn',
    'Gujarati': 'guj_Gujr',
    'Haitian': 'hat_Latn',
    'Hausa': 'hau_Latn',
    'Hebrew': 'heb_Hebr',
    'Hindi': 'hin_Deva',
    'Chhattisgarhi': 'hne_Deva',
    'Croatian': 'hrv_Latn',
    'Hungarian': 'hun_Latn',
    'Armenian': 'hye_Armn',
    'Igbo': 'ibo_Latn',
    'Ilocano': 'ilo_Latn',
    'Indonesian': 'ind_Latn',
    'Icelandic': 'isl_Latn',
    'Italian': 'ita_Latn',
    'Javanese': 'jav_Latn',
    'Japanese': 'jpn_Jpan',
    'Kabyle': 'kab_Latn',
    'Jingpho': 'kac_Latn',
    'Kamba': 'kam_Latn',
    'Kannada': 'kan_Knda',
    'Kashmiri (Arabic script)': 'kas_Arab',
    'Kashmiri': 'kas_Deva',
    'Georgian': 'kat_Geor',
    'Kanuri (Arabic script)': 'knc_Arab',
    'Kanuri': 'knc_Latn',
    'Kazakh': 'kaz_Cyrl',
    'Kabiyè': 'kbp_Latn',
    'Kabuverdianu': 'kea_Latn',
    'Khmer': 'khm_Khmr',
    'Kikuyu': 'kik_Latn',
    'Kinyarwanda': 'kin_Latn',
    'Kyrgyz': 'kir_Cyrl',
    'Kimbundu': 'kmb_Latn',
    'Kurdish': 'kmr_Latn',
    'Kikongo': 'kon_Latn',
    'Korean': 'kor_Hang',
    'Lao': 'lao_Laoo',
    'Ligurian': 'lij_Latn',
    'Limburgish': 'lim_Latn',
    'Lingala': 'lin_Latn',
    'Lithuanian': 'lit_Latn',
    'Lombard': 'lmo_Latn',
    'Latgalian': 'ltg_Latn',
    'Luxembourgish': 'ltz_Latn',
    'Luba-Kasai': 'lua_Latn',
    'Ganda': 'lug_Latn',
    'Luo': 'luo_Latn',
    'Mizo': 'lus_Latn',
    'Latvian': 'lvs_Latn',
    'Magahi': 'mag_Deva',
    'Maithili': 'mai_Deva',
    'Malayalam': 'mal_Mlym',
    'Marathi': 'mar_Deva',
    'Minangkabau (Arabic script)': 'min_Arab',
    'Minangkabau': 'min_Latn',
    'Macedonian': 'mkd_Cyrl',
    'Plateau Malagasy': 'plt_Latn',
    'Maltese': 'mlt_Latn',
    'Meitei': 'mni_Beng',
    'Mongolian': 'khk_Cyrl',
    'Mossi': 'mos_Latn',
    'Maori': 'mri_Latn',
    'Burmese': 'mya_Mymr',
    'Dutch': 'nld_Latn',
    'Norwegian': 'nno_Latn',
    'Norwegian Bokmål': 'nob_Latn',
    'Nepali': 'npi_Deva',
    'Northern Sotho': 'nso_Latn',
    'Nuer': 'nus_Latn',
    'Nyanja': 'nya_Latn',
    'Occitan': 'oci_Latn',
    'Oromo': 'gaz_Latn',
    'Odia': 'ory_Orya',
    'Pangasinan': 'pag_Latn',
    'Panjabi': 'pan_Guru',
    'Papiamento': 'pap_Latn',
    'Persian': 'pes_Arab',
    'Polish': 'pol_Latn',
    'Portuguese': 'por_Latn',
    'Dari': 'prs_Arab',
    'Pashto': 'pbt_Arab',
    'Quechua': 'quy_Latn',
    'Romanian': 'ron_Latn',
    'Rundi': 'run_Latn',
    'Russian': 'rus_Cyrl',
    'Sango': 'sag_Latn',
    'Sanskrit': 'san_Deva',
    'Santali': 'sat_Olck',
    'Sicilian': 'scn_Latn',
    'Shan': 'shn_Mymr',
    'Sinhala': 'sin_Sinh',
    'Slovak': 'slk_Latn',
    'Slovenian': 'slv_Latn',
    'Samoan': 'smo_Latn',
    'Shona': 'sna_Latn',
    'Sindhi': 'snd_Arab',
    'somali': 'som_Latn',
    'Sotho': 'sot_Latn',
    'Spanish': 'spa_Latn',
    'Tosk Albanian': 'als_Latn',
    'Sardinian': 'srd_Latn',
    'Serbian': 'srp_Cyrl',
    'Swati': 'ssw_Latn',
    'Sundanese': 'sun_Latn',
    'Swedish': 'swe_Latn',
    'Swahili': 'swh_Latn',
    'Silesian': 'szl_Latn',
    'Tamil': 'tam_Taml',
    'Tatar': 'tat_Cyrl',
    'Telugu': 'tel_Telu',
    'Tajik': 'tgk_Cyrl',
    'Tagalog': 'tgl_Latn',
    'Thai': 'tha_Thai',
    'Tigrinya': 'tir_Ethi',
    'Tamasheq': 'taq_Latn',
    'Tamasheq (Tifinagh script)': 'taq_Tfng',
    'Tok Pisin': 'tpi_Latn',
    'Tswana': 'tsn_Latn',
    'Tsonga': 'tso_Latn',
    'Turkmen': 'tuk_Latn',
    'Tumbuka': 'tum_Latn',
    'Turkish': 'tur_Latn',
    'Twi': 'twi_Latn',
    'Tamazight': 'tzm_Tfng',
    'Uyghur': 'uig_Arab',
    'Ukrainian': 'ukr_Cyrl',
    'Umbundu': 'umb_Latn',
    'Urdu': 'urd_Arab',
    'Northern Uzbek': 'uzn_Latn',
    'Venetian': 'vec_Latn',
    'Vietnamese': 'vie_Latn',
    'Waray': 'war_Latn',
    'Wolof': 'wol_Latn',
    'Xhosa': 'xho_Latn',
    'Yiddish': 'ydd_Hebr',
    'Yoruba': 'yor_Latn',
    'Cantonese': 'yue_Hant',
    'Chinese (Simplified)': 'zho_Hans',
    'Chinese': 'zho_Hant',
    'Malay': 'zsm_Latn',
    'Zulu': 'zul_Latn'}

language_codes_inv = {v: k for k, v in language_codes.items()}


################################################
####                datasets                ####
################################################

def get_translated_dataset_df(name,lang):
    """
    Loads a translated dataset from the directory in the requested language.
    
    Parameters:
    name: name of the dataset ['mgsm', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum']
    lang: language of the dataset to load.
    
    Returns:
    Dataset in the specified language as dataframe
    """
    lang = language_codes[lang]

    if name == "mgsm":
        
        df = pd.read_csv('./datasets/mgsm/mgsm_' + lang + '.csv',sep=';') 
        
        return df
    
    elif name == "xcopa":
       
        df = pd.read_csv('./datasets/xcopa/xcopa_' + lang + '.csv',sep=';') 
        
        return df
    
    elif name == 'msvamp':

        df = pd.read_csv('./datasets/msvamp/msvamp_' + lang + '.csv',sep=';') 

        return df
    
    elif name == 'coinflip':

        df = pd.read_csv('./datasets/coinflip/coinflip_' + lang + '.csv',sep=';') 

        return df
    
    elif name == 'shuffled_objects':

        df = pd.read_csv('./datasets/shuffled_objects/shuffled_objects_' + lang + '.csv',sep=';') 

        return df
    
    else:
        print("Dataset name is not correctly specified. Please input 'mgsm' or 'xcopa'.")

def get_dataset_df(name,lang):
    """
    Loads a test dataset from huggingface in the requested language.
    
    Parameters:
    name: name of the dataset ['mgsm', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum']
    lang: language of the dataset to load.
    
    Returns:
    Dataset in the specified language as dataframe.
    """
    
    if name == "mgsm":

        dataset = load_dataset("juletxara/mgsm",lang) 
        
        df = pd.DataFrame(data={'question' : dataset["test"]["question"],
                                'answer_number' : dataset["test"]["answer_number"]
                                })
        
        return df
    
    elif name == "xcopa" and lang == "en":

        dataset = load_dataset("pkavumba/balanced-copa")

        df = pd.DataFrame(data={'premise' : dataset["test"]["premise"],
                                'choice1' : dataset["test"]["choice1"],
                                'choice2' : dataset["test"]["choice2"],
                                'question' : dataset["test"]["question"],
                                'label' : dataset["test"]["label"]
                                })
        
        return df

    elif name == "xcopa":

        dataset = load_dataset("xcopa",lang) 
    
        df = pd.DataFrame(data={'premise' : dataset["test"]["premise"],
                                'choice1' : dataset["test"]["choice1"],
                                'choice2' : dataset["test"]["choice2"],
                                'question' : dataset["test"]["question"],
                                'label' : dataset["test"]["label"]
                                })
        
        return df

    elif name == "msvamp":

        dataset = load_dataset("Mathoctopus/MSVAMP",lang)

        df = pd.DataFrame(data={'m_query' : dataset["test"]["m_query"],
                                'response' : dataset["test"]["response"]
                                })
        
        return df
    
    elif name == "coinflip":
        
        dataset = pd.read_csv('./datasets/coinflip/coinflip_' + lang + '.csv', sep=';')
        
        return dataset 

    elif name == "shuffled_objects":

        dataset = pd.read_csv('./datasets/shuffled_objects/shuffled_objects_' + lang + '.csv', sep=';')
        
        return dataset
    
    else:
        print("Dataset name is not correctly specified. Please input 'mgsm', 'msvamp', 'xcopa', 'coinflip' or 'shuffled_objects'.")

# def get_exemplars_df(name,lang):
#     """
#     Loads a train dataset from huggingface in the requested language.
    
#     Parameters:
#     name: name of the dataset ['mgsm', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum']
#     lang: language of the dataset to load.
    
#     Returns:
#     Dataset in the specified language as dataframe.
#     """
#     dataset = get_dataset(name,lang)
    
#     if name == "mgsm":
#         df = pd.DataFrame(data={'question' : dataset["train"]["question"],
#                                 'answer' : dataset["train"]["answer"]
#                                 })
        
#         return df
    
#     elif name == "xcopa" and lang == "en":
#         df = pd.DataFrame(data={'premise' : dataset["train"]["premise"],
#                                 'choice1' : dataset["train"]["choice1"],
#                                 'choice2' : dataset["train"]["choice2"],
#                                 'question' : dataset["train"]["question"],
#                                 'label' : dataset["train"]["label"]
#                                 })
        
#         return df
    
#     elif name == "xcopa":
#         df = pd.DataFrame(data={'premise' : dataset["validation"]["premise"],
#                                 'choice1' : dataset["validation"]["choice1"],
#                                 'choice2' : dataset["validation"]["choice2"],
#                                 'question' : dataset["validation"]["question"],
#                                 'label' : dataset["validation"]["label"]
#                                 })
        
#         return df
    
    
#     else:
#         print("Dataset name is not correctly specified. Please input 'mgsm', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum'.")