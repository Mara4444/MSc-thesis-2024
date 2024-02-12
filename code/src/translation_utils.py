
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from datasets import load_dataset

nltk.download('punkt')

################################################
####            machine translation         ####
################################################

def translate_list(input_list,trg_lang,model,tokenizer):
    """
    Translate a list from English to the target language.
    
    Parameters:
    input_list: input list of strings to translate.
    trg_lang: target language given in iso2-code.
    
    Returns:
    Translated list of strings.
    """
    translated_list = []

    for string in input_list:
        
        translated_string = ""
        
        sentences = sent_tokenize(string)

        for sentence in sentences:
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
            print(sentence, translated_sentence)
            translated_string = translated_string + translated_sentence + ' '

        translated_list.append(translated_string)

    return translated_list

def translate_dataset(dataset,name,trg_lang,model,tokenizer): 
    """
    Translate a dataset from English to the target language.
    
    Parameters:
    dataset: dataset to translate.
    name: name of the dataset ['mgsm', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum']
    trg_lang: target language given in iso2-code.
    
    Returns:
    Translated dataset and returns as DataFrame. 
    """
    if name  == 'mgsm': 

        translated1_list = translate_list(dataset["test"]["question"],trg_lang,model,tokenizer)

        translated_dataset = pd.DataFrame({'question': translated1_list,
                                           'answer_number': dataset["test"]["answer_number"]
                                           })

        translated_dataset.to_csv('../datasets/mgsm/mgsm_' + trg_lang + '.csv', sep=';', index=False, header=True)
        # translated_dataset.to_csv('mgsm_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset
      
    elif name  == 'xcopa': 

        translated1_list = translate_list(dataset["test"]["premise"],trg_lang,model,tokenizer)
        translated2_list = translate_list(dataset["test"]["question"],trg_lang,model,tokenizer)
        translated3_list = translate_list(dataset["test"]["choice1"],trg_lang,model,tokenizer)
        translated4_list = translate_list(dataset["test"]["choice2"],trg_lang,model,tokenizer)

        translated_dataset = pd.DataFrame({'premise': translated1_list,
                                           'question': translated2_list,
                                           'choice1': translated3_list,
                                           'choice2': translated4_list, 
                                           'label': dataset["test"]["label"]
                                           })

        translated_dataset.to_csv('./datasets/xcopa/xcopa_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset
    
    elif name  == 'xstorycloze': 

        translated1_list = translate_list(dataset["eval"]["input_sentence_1"],trg_lang,model,tokenizer)
        translated2_list = translate_list(dataset["eval"]["input_sentence_2"],trg_lang,model,tokenizer)
        translated3_list = translate_list(dataset["eval"]["input_sentence_3"],trg_lang,model,tokenizer)
        translated4_list = translate_list(dataset["eval"]["input_sentence_4"],trg_lang,model,tokenizer)
        translated5_list = translate_list(dataset["eval"]["sentence_quiz1"],trg_lang,model,tokenizer)
        translated6_list = translate_list(dataset["eval"]["sentence_quiz2"],trg_lang,model,tokenizer)
        
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
        
        translated1_list = translate_list(dataset["test"]["sentence1"],trg_lang,model,tokenizer)
        translated2_list = translate_list(dataset["test"]["sentence2"],trg_lang,model,tokenizer)

        translated_dataset = pd.DataFrame({'sentence1': translated1_list,
                                           'sentence2': translated2_list,
                                           'label': dataset["test"]["label"]
                                           })

        translated_dataset.to_csv('./datasets/pawsx/pawsx_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset
    
    elif name  == 'xnli': 
        
        translated1_list = translate_list(dataset["test"]["premise"],trg_lang,model,tokenizer)
        translated2_list = translate_list(dataset["test"]["hypothesis"],trg_lang,model,tokenizer)

        translated_dataset = pd.DataFrame({'premise': translated1_list,
                                           'hypothesis': translated2_list,
                                           'label': dataset["test"]["label"]
                                           })

        translated_dataset.to_csv('./datasets/xnli/xnli_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset
    
    elif name  == 'xlsum': 
        
        translated1_list = translate_list(dataset["test"]["title"],trg_lang,model,tokenizer)
        translated2_list = translate_list(dataset["test"]["summary"],trg_lang,model,tokenizer)
        translated3_list = translate_list(dataset["test"]["text"],trg_lang,model,tokenizer)

        translated_dataset = pd.DataFrame({'title': translated1_list,
                                           'summary': translated2_list,
                                           'text': translated3_list
                                           })

        translated_dataset.to_csv('./datasets/xlsum/xlsum_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset

    else:
        print("Dataset name is not correctly specified. Please input 'mgsm', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum'.")

def translate_exemplars(dataset,languages,model,tokenizer):
    """
    Translate the exemplars of a dataset from English to the target language.
    
    Parameters:
    dataset: dataset to translate.
    langauges: list of languages available in the nllb model to translate to.
    
    Returns:
    Translated dataset with exemplars and returns as DataFrame. 
    """
    lang_list = []
    question_list = []
    answer_list = []

    for lang in languages:
        # translate exemplars
        translated1_list = translate_list(dataset["train"]["question"],lang,model,tokenizer)
        translated2_list = translate_list(dataset["train"]["answer"],lang,model,tokenizer)

        for i in [lang]*len(dataset["train"]["question"]):
            lang_list.append(i)
        for j in translated1_list:
            question_list.append(j)
        for k in translated2_list:
            answer_list.append(k)

    translated_exemplars = pd.DataFrame({'language' : lang_list,
                                        'question': question_list,
                                        'answer': answer_list
                                        })

    translated_exemplars.to_csv('mgsm_translated_exemplars_llama.csv', sep=';', index=False, header=True)

    df = pd.read_csv('../datasets/mgsm/mgsm_exemplars_original.csv',sep=';')
    merged_df = pd.concat([df, translated_exemplars])

    merged_df.to_csv('mgsm_exemplars_llama.csv', sep=';', index=False, header=True)

    return translated_exemplars 