from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import torch.nn as nn
import pandas as pd
import random
import re
import numpy as np

################################################
####          chain-of-thougth              ####
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
                  'Serbian': 'srp_Cyrl',
                  'Spanish' : 'spa_Latn',
                  'Swedish': 'swe_Latn',
                  'Ukrainian': 'ukr_Cyrl',
                  'Vietnamese': 'vie_Latn'}

def get_exemplars(nr_shots,shots_lang):
    """
    Generate a list of n exemplar strings in the targeted language.
    
    Parameters:
    nr_shots: nr. of exemplars to select.
    shots_lang: language of the exemplars.

    Returns:
    String with all exemplars.
    """
    shots_lang = language_codes[shots_lang]

    if nr_shots > 0:
        exemplars = pd.read_csv('datasets/mgsm/mgsm_exemplars_llama.csv', sep=';')
        exemplars = exemplars[exemplars['language'] == shots_lang] # select target language exemplars

        exemplar_string = ''
        if 0 <= nr_shots <= len(exemplars):
            sampled_exemplars = exemplars.sample(n=nr_shots, random_state=2024)
            
            for _, row in sampled_exemplars.iterrows():
                exemplar_string += ' ' + row.iloc[1] + ' ' + row.iloc[2]

            return exemplar_string + ' '
        
        else:
            print('The nr_shots input is not correctly specified. The maximum number of exemplars is ', len(exemplars))
    
    else:
        return ''

def get_prompt(question,prompt_setting,cot_lang):
    """
    Generate a string response by a prompt and promptsetting.
    
    Parameters:
    question: string task.
    prompt_setting: different prompting techniques: 'basic', 'cot'
    cot_lang: language to perform the cot reasoning in.

    Returns:
    String prompt including the prompting method.
    """

    if prompt_setting == 'basic':
        return question
    
    elif prompt_setting == 'cot':
        return question + " Let's think step by step in " + cot_lang + "."
        
    else:
        print("Prompt setting not correctly specified. Please indicate 'basic' or 'cot'.")

def mgsm_generate_response(df,task_lang,prompt_setting,nr_shots,shots_lang,cot_lang,model,tokenizer,name):
    """
    Generate a text response by a given LLM for prompts in a list.
    
    Parameters:
    df: dataframe with questions and answers of the mgsm benchmark.
    task_lang: the language of the prompts in the dataset.
    cot_lang: the required language for the reasoning steps.
    prompt_setting: different prompting techniques: 'basic', 'cot'. 
    model: initialized model.
    tokenizer: initializer tokenizer.
    nr_shots: number of exemplars to select.
    shots_lang: language of the exemplars.
    
    Returns:
    Text generated respons by the LLM for each prompt in the list.
    """
    questionlist = df.iloc[:,0].tolist()
    
    pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    )

    responselist = []

    for question in questionlist:
        sequences = pipeline(
        get_exemplars(nr_shots,shots_lang) + get_prompt(question,prompt_setting,cot_lang),
        do_sample=False, # greedy approach
        # temperature=0.0, # greedy approach
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1000, # max 6 exemplars + question?
        return_full_text=False,
        )
        
        for seq in sequences:
            print(get_exemplars(nr_shots,shots_lang) + get_prompt(question,prompt_setting,cot_lang))
            print(f"Response: {seq['generated_text']}")
            responselist.append(f"Response: {seq['generated_text']}")
    # print(responselist)
    response = pd.DataFrame(data=[row.split(sep=", '") for row in responselist]) # converting the translated questionlist to a pandas df

    if prompt_setting == 'basic':
        title = name + '_mgsm_' + task_lang + '_' + prompt_setting + '_' + str(nr_shots) + '-shot_' + shots_lang + '.csv'
        response.to_csv('results/' + title, sep=';', index=False, header=False)
        print(title, ' saved.')

    elif prompt_setting == 'cot':
        title = name + '_mgsm_' + task_lang + '_' + prompt_setting + '_' + cot_lang + '_' + str(nr_shots) + '-shot_' + shots_lang + '.csv'
        response.to_csv('results/' + title, sep=';', index=False, header=False)
        print(title, ' saved.')
    
    
def extract_answer(inputstring):
    """
    Finds the last mentioned number in a string.
    
    Parameters:
    input: string.

    Returns:
    String value of the last mentioned number.
    """
    # Regular expression to find 'the answer is ' followed by a number
    match = re.search(r'the answer is (\b\d+(?:[,.]\d+)?\b)', inputstring,re.IGNORECASE)

    if match:
        print('match found')
        # Extract the number after 'the answer is'
        number = match.group(1)
        return number
    else:
        numberlist = re.findall(r'\b\d+(?:[,.]\d+)?\b',inputstring)
        print('take last nr')
        if len(numberlist) != 0:
            return numberlist[-1]
        else:
            return None

def calculate_accuracy(df1,df2):
    """
    Calculate the accuracy (% correct answers) from two input tsv files.
    
    Parameters:
    df1: orginial mgsm English file with correct answer column.
    df2: response mgsm file with predicted answer column.

    Returns:
    Accuracy score (% of correct answers).
    """
    correct_answerlist = df1.iloc[:,1].tolist()
    predicted_answerlist = df2.iloc[:,1].tolist()

    total = len(correct_answerlist)
    # print("Total answers: ",total)
    
    nr_correct = 0

    for answer in correct_answerlist:
        loc = correct_answerlist.index(answer)
        if predicted_answerlist[loc] == answer:
            nr_correct = nr_correct + 1
    
    # print("Correct answers: ", nr_correct)
    accuracy = round(100*(nr_correct / total),1)

    return accuracy