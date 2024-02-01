
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# download machine translation model
model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

################################################
####            machine translation         ####
################################################

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
            print(sentence)
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
            print(translated_sentence)
            translated_string = translated_string + translated_sentence + ' '
        # print(translated_string)
        translated_list.append(translated_string)
        # print(translated_list)
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

        translated_dataset.to_csv('./datasets/mgsm/mgsm_' + trg_lang + '.csv', sep=';', index=False, header=True)

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

        translated_dataset.to_csv('./datasets/xcopa/xcopa_' + trg_lang + '.csv', sep=';', index=False, header=True)

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

        translated_dataset.to_csv('./datasets/xstorycloze/xstorycloze_' + trg_lang + '.csv', sep=';', index=False, header=True)

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

        translated_dataset.to_csv('./datasets/pawsx/pawsx_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset
    
    elif name  == 'xnli': 
        
        translated1_list = translate_list(dataset["test"]["premise"],src_lang,trg_lang)
        translated2_list = translate_list(dataset["test"]["hypothesis"],src_lang,trg_lang)

        translated_dataset = pd.DataFrame({'premise': translated1_list,
                                           'hypothesis': translated2_list,
                                           'label': dataset["test"]["label"]
                                           })

        translated_dataset.to_csv('./datasets/xnli/xnli_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset
    
    elif name  == 'xlsum': 
        
        translated1_list = translate_list(dataset["test"]["title"],src_lang,trg_lang)
        translated2_list = translate_list(dataset["test"]["summary"],src_lang,trg_lang)
        translated3_list = translate_list(dataset["test"]["text"],src_lang,trg_lang)

        translated_dataset = pd.DataFrame({'title': translated1_list,
                                           'summary': translated2_list,
                                           'text': translated3_list
                                           })

        translated_dataset.to_csv('./datasets/xlsum/xlsum_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset

    else:
        print("Dataset name is not correctly specified. Please input 'mgsm', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum'.")

def translate_exemplars(dataset,name,src_lang,trg_lang):
    
    if name == "mgsm":
        # translate exemplars
        translated2_list = translate_list(dataset["train"]["question"],src_lang,trg_lang)
        translated3_list = translate_list(dataset["train"]["answer"],src_lang,trg_lang)

        translated_exemplars = pd.DataFrame({'question': translated2_list,
                                            'answer': translated3_list
                                            })

        translated_exemplars.to_csv('./datasets/mgsm/mgsm_' + trg_lang + 'exemplars.csv', sep=';', index=False, header=True)

        return translated_exemplars 
    
    else:
        print("Dataset name is not correctly specified. Please input 'mgsm'")