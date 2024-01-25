from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from datasets import load_dataset
import pandas as pd

# download model
model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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
    translated_list = []
    translator = pipeline(
        'translation', 
        model=model, 
        tokenizer=tokenizer
        )
    
    for string in input_list:
        output = translator(string, 
                            src_lang=src_lang, 
                            tgt_lang=trg_lang
                            )
        translated_text = output[0]['translation_text']
        print(translated_text)
        translated_list.append(translated_text)
        # translated_list.append('test')
    
    return translated_list

input_list = ['This is the sentence to translate.']
translate_list(input_list,"en","nl")