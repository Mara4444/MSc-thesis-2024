
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
    trg_lang: target language given in nllb-200 code.
    
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

        return translated_dataset
    
    elif name  == 'xcopa': 

        translated1_list = translate_list(dataset["test"]["premise"],trg_lang,model,tokenizer)
        # translated2_list = translate_list(dataset["test"]["question"],trg_lang,model,tokenizer)
        translated3_list = translate_list(dataset["test"]["choice1"],trg_lang,model,tokenizer)
        translated4_list = translate_list(dataset["test"]["choice2"],trg_lang,model,tokenizer)

        translated_dataset = pd.DataFrame({'premise': translated1_list,
                                           'question': dataset["test"]["question"],
                                           'choice1': translated3_list,
                                           'choice2': translated4_list, 
                                           'label': dataset["test"]["label"]
                                           })

        translated_dataset.to_csv('../datasets/xcopa/xcopa_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset
    
    elif name == "coinflip":

        translated1_list = translate_list(dataset["question"],trg_lang,model,tokenizer)
        translated2_list = translate_list(dataset["answer"],trg_lang,model,tokenizer)

        translated_dataset = pd.DataFrame({'question': translated1_list,
                                           'answer': translated2_list,
                                           'answer_ab': dataset["answer_ab"]
                                           })

        translated_dataset.to_csv('../datasets/coinflip/coinflip_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset
    
    elif name == "shuffled_objects":

        translated1_list = translate_list(dataset["inputs"],trg_lang,model,tokenizer)
        translated2_list = translate_list(dataset["A"],trg_lang,model,tokenizer)
        translated3_list = translate_list(dataset["B"],trg_lang,model,tokenizer)
        translated4_list = translate_list(dataset["C"],trg_lang,model,tokenizer)
        translated5_list = translate_list(dataset["answer"],trg_lang,model,tokenizer)

        translated_dataset = pd.DataFrame({'question': translated1_list,
                                           'A': translated2_list,
                                           'B': translated3_list,
                                           'C': translated4_list,
                                           'answer' : translated5_list,
                                           'answer_abc': dataset["answer_abc"]
                                           })

        translated_dataset.to_csv('../datasets/shuffled_objects/shuffled_objects_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset
    
    elif name  == 'msvamp': 

        translated1_list = translate_list(dataset["m_query"],trg_lang,model,tokenizer)

        translated_dataset = pd.DataFrame({'m_query': translated1_list,
                                           'response': dataset["response"]
                                           })

        translated_dataset.to_csv('../datasets/msvamp/msvamp_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset

    else:
        print("Dataset name is not correctly specified.")

def translate_instruction_cot(languages,model,tokenizer):
    """
    Translate the "Let's think step by step." statement to the target language.
    
    Parameters:
    languages: list of languages available in the nllb model to translate to.
    
    Returns:
    Translated dataset with cot statement and returns as DataFrame. 
    """
    translated_list = []

    for lang in languages:
        
        translated_list.append(translate_string(inputstring="Let's think step by step.",trg_lang=lang,model=model,tokenizer=tokenizer) )

    translated_instructions = pd.DataFrame({'language' : languages,
                                   'cot' : translated_list,
                                   })

    translated_instructions.to_csv('translated_instructions_cot.csv', sep=';', index=False, header=True)

    return translated_instructions

def translate_instruction_xcopa(languages,model,tokenizer):
    """
    Translate the prompt instruction of XCOPA to the target languages.
    
    Parameters:
    languages: list of languages available in the nllb model to translate to.
    
    Returns:
    Translated dataset with XCOPA prompt instruction and returns as DataFrame. 
    """
    translated_list1 = []
    translated_list2 = []

    for lang in languages:

        instruction1 = translate_string(inputstring="Premise:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction2 = translate_string(inputstring="Option A:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction3 = translate_string(inputstring="Option B:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction4 = translate_string(inputstring="Based on the premise, which cause is more likely?",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction5 = translate_string(inputstring="Based on the premise, which effect is more likely?",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction6 = translate_string(inputstring="Pick between options A and B.",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction7 = translate_string(inputstring="Answer:",trg_lang=lang,model=model,tokenizer=tokenizer) 

        instruction_cause = instruction1 + ' {premise} \n' + instruction2 + ' {choice1} \n' + instruction3 + '{choice2} \n' + instruction4 + '\n' + instruction6 + '\n' + instruction7 + ' {cot}'
        instruction_effect = instruction1 + ' {premise} \n' + instruction2 + ' {choice1} \n' + instruction3 + '{choice2} \n' + instruction5 + '\n' + instruction6 + '\n' + instruction7 + ' {cot}'
        
        translated_list1.append(instruction_cause)
        translated_list2.append(instruction_effect)

    translated_instructions = pd.DataFrame({'language' : languages,
                                   'xcopa_cause' : translated_list1,
                                   'xcopa_effect' : translated_list2
                                   })

    translated_instructions.to_csv('translated_instructions_xcopa.csv', sep=';', index=False, header=True)

    return translated_instructions

def translate_instruction_mgsm(languages,model,tokenizer):
    """
    Translate the prompt instruction of MGSM to the target languages.
    
    Parameters:
    languages: list of languages available in the nllb model to translate to.
    
    Returns:
    Translated dataset with MGSM prompt instruction and returns as DataFrame. 
    """
    translated_list1 = []
    translated_list2 = []

    for lang in languages:

        instruction1 = translate_string(inputstring="Question:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction2 = translate_string(inputstring="Based on the question, formulate a numeric answer.",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction3 = translate_string(inputstring="Answer:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction4 = translate_string(inputstring="Let's think step by step.",trg_lang=lang,model=model,tokenizer=tokenizer)

        instruction_basic = instruction1 + ' {question} \n' + instruction2 + ' ' + instruction3
        instruction_cot = instruction1 + ' {question} \n' + instruction2 + ' ' + instruction3 + ' ' + instruction4
        
        translated_list1.append(instruction_basic)
        translated_list2.append(instruction_cot)

    translated_instructions = pd.DataFrame({'language' : languages,
                                   'mgsm_basic' : translated_list1,
                                   'mgsm_cot' : translated_list2
                                   })

    translated_instructions.to_csv('translated_instructions_mgsm.csv', sep=';', index=False, header=True)

    return translated_instructions

def translate_instruction_coinflip(languages,model,tokenizer):
    """
    Translate the prompt instruction of coinflip to the target languages.
    
    Parameters:
    languages: list of languages available in the nllb model to translate to.
    
    Returns:
    Translated dataset with coinflip prompt instruction and returns as DataFrame. 
    """
    translated_list1 = []
    translated_list2 = []

    for lang in languages:

        instruction1 = translate_string(inputstring="Question:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction2 = translate_string(inputstring="Note that 'flip' here means 'reverse'.",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction3 = translate_string(inputstring="Option A: Yes",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction4 = translate_string(inputstring="Option B: No",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction5 = translate_string(inputstring="Pick between options A and B.",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction6 = translate_string(inputstring="Answer: ",trg_lang=lang,model=model,tokenizer=tokenizer) 

        instruction = instruction1 + ' {question} ' + instruction2 + ' \n' + instruction3 + ' \n' + instruction4 + ' \n' + instruction5 + ' \n' + instruction6
                
        translated_list1.append(instruction)

    translated_instructions = pd.DataFrame({'language' : languages,
                                   'coinflip' : translated_list1})

    translated_instructions.to_csv('translated_instructions_coinflip.csv', sep=';', index=False, header=True)

    return translated_instructions
        
        
def translate_string(inputstring,trg_lang,model,tokenizer):
    """
    Translate a string from English to the target language.
    
    Parameters:
    inputstringt: input string to translate.
    trg_lang: target language given in nllb-200 code.
    
    Returns:
    Translated string.
    """
    
    translated_string = ""
        
    sentences = sent_tokenize(inputstring)

    for sentence in sentences:
        inputs = tokenizer(
            sentence, 
            return_tensors="pt"
            )
        
        translated_tokens = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.lang_code_to_id[trg_lang], 
            max_length=100
            )
            
        translated_sentence = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        print(sentence, translated_sentence)
        translated_string = translated_string + translated_sentence + ' '

    return translated_string


# def translate_exemplars(dataset,languages,model,tokenizer):
#     """
#     Translate the exemplars of a dataset from English to the target language.
    
#     Parameters:
#     dataset: dataset to translate.
#     langauges: list of languages available in the nllb model to translate to.
    
#     Returns:
#     Translated dataset with exemplars and returns as DataFrame. 
#     """
#     lang_list = []
#     question_list = []
#     answer_list = []

#     for lang in languages:
#         # translate exemplars
#         translated1_list = translate_list(dataset["train"]["question"],lang,model,tokenizer)
#         translated2_list = translate_list(dataset["train"]["answer"],lang,model,tokenizer)

#         for i in [lang]*len(dataset["train"]["question"]):
#             lang_list.append(i)
#         for j in translated1_list:
#             question_list.append(j)
#         for k in translated2_list:
#             answer_list.append(k)

#     translated_exemplars = pd.DataFrame({'language' : lang_list,
#                                         'question': question_list,
#                                         'answer': answer_list
#                                         })

#     translated_exemplars.to_csv('mgsm_translated_exemplars_llama.csv', sep=';', index=False, header=True)

#     df = pd.read_csv('../datasets/mgsm/mgsm_exemplars_original.csv',sep=';')
#     merged_df = pd.concat([df, translated_exemplars])

#     merged_df.to_csv('mgsm_exemplars_llama.csv', sep=';', index=False, header=True)

#     return translated_exemplars 