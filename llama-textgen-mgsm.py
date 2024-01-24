from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM, pipeline
from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
import torch
import torch.nn as nn
import pandas as pd
import re

######################### todo #############################
# - decide on prompt_setting
# - create function that extracts the answer from the response
# - create function that calculates the accuracy
# - create function that <after mgsm_generate_response> makes sure that the response df has 1 response string per row and extracts the answer for each response and puts this as new column in df
# - run the function for each mgsm task_lang + cot_lang combination > llama-textgen-mgsm-batch.py


# Llama-2 model
# model_name = "/gpfs/home1/msmeets/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/" # regular model
model_name = "/gpfs/home1/msmeets/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/" # chat model

tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# settings
prompt_setting = "basic"
cot_lang = "English"
loc = "/gpfs/home1/msmeets/mgsm/mgsm_French.tsv"


dataset = pd.read_csv(loc,sep="\t",header=None)
task_lang = loc[30:-4]

def get_prompt(prompt,prompt_setting):
    """
    Generate a string response by a prompt and promptsetting.
    
    Parameters:
    prompt: string task.
    prompt_setting: different prompting techniques: 'basic', 'huang', 'upadhayay', 'qin_cla', 'qin_tss'

    Returns:
    String prompt including the prompting method.
    """
    if prompt_setting == 'basic':
        return prompt + " Let's think step by step in " + cot_lang + "."
    
    elif prompt_setting == 'huang':
        return prompt + " I want you to act as an arithmic expert for " + task_lang + ". You should retell the request in " + cot_lang + ". You should do step-by-step answer to obtain a number answer. You should step-by-step answer the request. You should tell me the answer in this format 'Answer:'."
    
    elif prompt_setting == 'upadhayay':
        return " Translate the following instructions from " + task_lang + " to " + cot_lang + ", formulate a response in " + cot_lang + ", and then translate that response back into " + task_lang + "." + prompt
    
    elif prompt_setting == 'qin_cla':
        return prompt + " After understanding, you should act as an expert in arithmetic reasoning in " + cot_lang + ". Let's resolve the task you understand above step-by-step! Finally, you should format your answer as 'Answer: [num]'"
    
    elif prompt_setting == 'qin_tss':
        return " Please act as an expert in multi-lingual understanding in " + task_lang + ". Request: " + prompt + " Let's understand the task in " + cot_lang + " step-by-step!"
    
    else:
        print("Prompt setting not correctly specified.")


def mgsm_generate_response(dataset,task_lang,cot_lang,prompt_setting,model,tokenizer):
    """
    Generate a text response by a given LLM for prompts in a list.
    
    Parameters:
    dataset: list of strings with prompts to feed the LLM.
    task_lang: the language of the prompts in the dataset.
    cot_lang: the required language for the reasoning steps.
    prompt_setting: different prompting techniques: 'basic', 'huang', 'upadhayay', 'qin_cla', 'qin_tss'
    model: initialized model.
    tokenizer: initializer tokenizer.
    
    Returns:
    Text generated respons by the LLM for each prompt in the list.
    """
    promptlist = dataset.iloc[:3,0].tolist() # questionlist from mgsm ([:3,0] for testing first 5 rows)
    
    pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    )

    responselist = []

    for prompt in promptlist:
        sequences = pipeline(
        get_prompt(prompt,prompt_setting),
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=500,
        return_full_text=False,
        )

        for seq in sequences:
            print(f"Response: {seq['generated_text']}")
            responselist.append(f"Response: {seq['generated_text']}")

    response = pd.DataFrame(data=[row.split(sep="',") for row in responselist]) # converting the translated questionlist to a pandas df
    response.to_csv('response_mgsm_' + task_lang + '_cot_' + prompt_setting + '_' + cot_lang + '.tsv', sep='\t', index=False, header=False)

mgsm_generate_response(dataset,task_lang,cot_lang,prompt_setting,model,tokenizer)

def extract_answer(inputstring):
    """
    Finds the last mentioned number in a string.
    
    Parameters:
    input: string.

    Returns:
    String value of the last mentioned number.
    """
    numberlist = re.findall(r'\b\d+(?:[,.]\d+)?\b',inputstring)

    if len(numberlist) != 0:
        return numberlist[-1]
    else:
        return 'NaN'

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