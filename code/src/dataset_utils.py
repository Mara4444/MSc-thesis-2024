
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datasets import load_dataset
import pandas as pd

################################################
####                datasets                ####
################################################

language_codes = {'Bulgarian': 'bul_Cyrl',
                  'Catalan': 'cat_Latn',
                  'Chinese' : 'zho_Hant',
                  'Croatian': 'hrv_Latn',
                  'Czech': 'ces_Latn',
                  'Danish': 'dan_Latn',
                  'Dutch': 'nld_Latn',
                  'English': 'eng_Latn',
                  'Finnish': 'fin_Latn',
                  'French' : 'fre_Latn',
                  'German' : 'deu_Latn',
                  'Hungarian': 'hun_Latn',
                  'Indonesian': 'ind_Latn',
                  'Italian': 'ita_Latn',
                  'Japanese' : 'Jpan',
                  'Korean': 'kor_Hang',
                  'Norwegian': 'nno_Latn',
                  'Polish': 'pol_Latn',
                  'Portuguese': 'por_Latn',
                  'Romanian': 'ron_Latn',
                  'Russian' : 'rus_Cyrl',
                  'Slovenian': 'slv_Latn',
                  'Spanish' : 'spa_Latn',
                  'Serbian': 'srp_Cyrl',
                  'Swedish': 'swe_Latn',
                  'Ukrainian': 'ukr_Cyrl',
                  'Vietnamese': 'vie_Latn'}

def get_dataset(name,lang):
    """
    Loads a dataset from huggingface in the requested language.
    
    Parameters:
    name: name of the dataset ['mgsm', 'msvamp', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum']
    lang: language of the dataset to load.
    
    Returns:
    Dataset in the specified language.
    """
    if name == "mgsm":
        dataset = load_dataset("juletxara/mgsm",lang) 
        
        return dataset
    
    elif name == "msvamp":
        dataset = load_dataset("Mathoctopus/MSVAMP",lang)

        return dataset
    
    # elif name == "xcopa" and lang == "en":
    #     dataset = load_dataset("pkavumba/balanced-copa")
        
    #     return dataset
    
    # elif name == "xcopa":
    #     dataset = load_dataset("xcopa",lang) 
        
    #     return dataset 
    
    # elif name == "xstorycloze":
    #     dataset = load_dataset("juletxara/xstory_cloze",lang)   
        
    #     return dataset
    
    # elif name == "mkqa":
    #     dataset = load_dataset("mkqa")
        
    #     if lang in dataset["train"]["queries"][0].keys():
    #         questionlist = [language[lang] for language in dataset["train"]["queries"]]
    #         answerlist = [language[lang] for language in dataset["train"]["answers"]]
            
    #         dataset = {
    #             "train": {
    #                 "queries": questionlist,
    #                 "answers": answerlist,
    #             }
    #         }

    #         return dataset
        
        # else:
        #     print("Language not found.")
    
    # elif name == "pawsx":
    #     dataset = load_dataset("paws-x",lang)    

    #     return dataset
    
    # elif name == "xnli":
    #     dataset = load_dataset("xnli",lang)  

    #     return dataset
    
    # elif name == "xlsum":        
    #     dataset = load_dataset("csebuetnlp/xlsum",lang) # ['amharic', 'arabic', 'azerbaijani', 'bengali', 'burmese', 'chinese_simplified', 'chinese_traditional', 'english', 'french', 'gujarati', 'hausa', 'hindi', 'igbo', 'indonesian', 'japanese', 'kirundi', 'korean', 'kyrgyz', 'marathi', 'nepali', 'oromo', 'pashto', 'persian', 'pidgin', 'portuguese', 'punjabi', 'russian', 'scottish_gaelic', 'serbian_cyrillic', 'serbian_latin', 'sinhala', 'somali', 'spanish', 'swahili', 'tamil', 'telugu', 'thai', 'tigrinya', 'turkish', 'ukrainian', 'urdu', 'uzbek', 'vietnamese', 'welsh', 'yoruba']

    #     return dataset
    
    else:
        print("Dataset name is not correctly specified. Please input 'mgsm', 'msvamp 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum'.")

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
    
    else:
        print("Dataset name is not correctly specified. Please input 'mgsm'.")

def get_dataset_df(name,lang):
    """
    Loads a test dataset from huggingface in the requested language.
    
    Parameters:
    name: name of the dataset ['mgsm', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum']
    lang: language of the dataset to load.
    
    Returns:
    Dataset in the specified language as dataframe.
    """
    dataset = get_dataset(name,lang)
    
    if name == "mgsm":
        df = pd.DataFrame(data={'question' : dataset["test"]["question"],
                                'answer_number' : dataset["test"]["answer_number"]
                                })
        
        return df
    
    elif name == "msvamp":
        df = pd.DataFrame(data={'m_query' : dataset["test"]["m_query"],
                                'response' : dataset["test"]["response"]
                                })
        
        return df
    
    elif name == "xcopa" and lang == "en":
        df = pd.DataFrame(data={'premise' : dataset["test"]["premise"],
                                'choice1' : dataset["test"]["choice1"],
                                'choice2' : dataset["test"]["choice2"],
                                'question' : dataset["test"]["question"],
                                'label' : dataset["test"]["label"]
                                })
        
        return df
    
    
    elif name == "xcopa":
        df = pd.DataFrame(data={'premise' : dataset["test"]["premise"],
                                'choice1' : dataset["test"]["choice1"],
                                'choice2' : dataset["test"]["choice2"],
                                'question' : dataset["test"]["question"],
                                'label' : dataset["test"]["label"]
                                })
        
        return df
    
    ## add other datasets also
    
    else:
        print("Dataset name is not correctly specified. Please input 'mgsm', 'msvamp', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum'.")

def get_exemplars_df(name,lang):
    """
    Loads a train dataset from huggingface in the requested language.
    
    Parameters:
    name: name of the dataset ['mgsm', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum']
    lang: language of the dataset to load.
    
    Returns:
    Dataset in the specified language as dataframe.
    """
    dataset = get_dataset(name,lang)
    
    if name == "mgsm":
        df = pd.DataFrame(data={'question' : dataset["train"]["question"],
                                'answer' : dataset["train"]["answer"]
                                })
        
        return df
    
    elif name == "xcopa" and lang == "en":
        df = pd.DataFrame(data={'premise' : dataset["train"]["premise"],
                                'choice1' : dataset["train"]["choice1"],
                                'choice2' : dataset["train"]["choice2"],
                                'question' : dataset["train"]["question"],
                                'label' : dataset["train"]["label"]
                                })
        
        return df
    
    elif name == "xcopa":
        df = pd.DataFrame(data={'premise' : dataset["validation"]["premise"],
                                'choice1' : dataset["validation"]["choice1"],
                                'choice2' : dataset["validation"]["choice2"],
                                'question' : dataset["validation"]["question"],
                                'label' : dataset["validation"]["label"]
                                })
        
        return df
    
    ## add other datasets also
    
    else:
        print("Dataset name is not correctly specified. Please input 'mgsm', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum'.")