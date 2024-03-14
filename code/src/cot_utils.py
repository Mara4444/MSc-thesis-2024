from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, BloomTokenizerFast, BloomForCausalLM
import transformers
import torch
import torch.nn as nn
import pandas as pd
import random
import re
import numpy as np

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

def get_prompt(row,task,prompt_setting,instr_lang):
    """
    Generate a string response by a prompt and promptsetting.
    
    Parameters:
    question: string task.
    prompt_setting: different prompting techniques: 'basic', 'cot'
    instr_lang: language of the instruction.

    Returns:    
    String prompt.
    """

    instr_lang = language_codes[instr_lang]

    def generate_message(string,**kwargs):
        return string.format(**kwargs)
    
    instructions = pd.read_csv("./datasets/translated_instructions.csv",sep=';')
    instructions.set_index(instructions['language'],inplace=True)
    instructions = instructions.drop('Unnamed: 0',axis=1)
    instructions = instructions.drop('language',axis=1)

    if task == 'xcopa':

        if prompt_setting == 'cot':

            cot = instructions.loc[instr_lang]['cot']

        elif prompt_setting == 'basic':

            cot = ''

        question = row['question']
        
        if question == 'cause':

            return generate_message(instructions.loc[instr_lang]['xcopa_cause'],
                            premise = row['premise'],
                            choice1 = row['choice1'],
                            choice2 = row['choice2'],
                            question = row['question'],
                            cot=cot)
            
        elif question == 'effect':

            return generate_message(instructions.loc[instr_lang]['xcopa_effect'],
                            premise = row['premise'],
                            choice1 = row['choice1'],
                            choice2 = row['choice2'],
                            question = row['question'],
                            cot=cot)
    
    # mgsm, msvamp, coinflip en shuffled objects nog niet van machine translated instructions


    elif task == 'mgsm':

        if prompt_setting == 'cot':

            return generate_message("Question: {question} \nBased on the question, formulate a numeric answer. \nAnswer: Let's think step by step.",
                                    question = row['question'])
        
            # return generate_message(instructions.loc[instr_lang]['mgsm_cot'],
            #                     question = row['question'])
        
        elif prompt_setting == 'basic':
            
            return generate_message("Question: {question} \nBased on the question, formulate a numeric answer. \nAnswer: ",
                                    question = row['question'])
            # return generate_message(instructions.loc[instr_lang]['mgsm_basic'],
            #                     question = row['question'])  
        
    elif task == 'msvamp':

        if prompt_setting == 'cot':

            return generate_message("Question: {question} \nBased on the question, formulate a numeric answer. \nAnswer: Let's think step by step.",
                                    question = row['m_query'])
        
            # return generate_message(instructions.loc[instr_lang]['mgsm_cot'],
            #                     question = row['question'])
        
        elif prompt_setting == 'basic':

            return generate_message("Question: {question} \nBased on the question, formulate a numeric answer. \nAnswer: ",
                                    question = row['m_query'])

            # return generate_message(instructions.loc[instr_lang]['mgsm_basic'],
            #                     question = row['question'])   
        
    elif task == 'coinflip':

        if prompt_setting == 'cot':
            
            return generate_message("Question: {question} \nOption A: Yes \nOption B: No \nBased on the question, which option is true? \nPick between options A and B. \nAnswer: Let's think step by step.",
                                    question = row['question'])
        
        elif prompt_setting == 'basic':

            return generate_message('Question: {question} \nOption A: Yes \nOption B: No \nBased on the question, which option is true? \nPick between options A and B. \nAnswer: ',
                                    question = row['question'])         
          
    elif task == 'shuffled_objects':

        if prompt_setting == 'cot':

            return generate_message("Question: {question} \nOption A: {a} \nOption B: {b} \nOption C: {c} \nBased on the question, which option is true? \nPick between options A, B and C. \nAnswer: Let's think step by step.",
                                    question = row['input'],
                                    a = row['A'],
                                    b = row['B'],
                                    c = row['C'])

        elif prompt_setting == 'basic':
            
            return generate_message("Question: {question} \nOption A: {a} \nOption B: {b} \nOption C: {c} \nBased on the question, which option is true? \nPick between options A, B and C. \nAnswer: ",
                                    question = row['input'],
                                    a = row['A'],
                                    b = row['B'],
                                    c = row['C'])


def generate_response(df,task,task_lang,instr_lang,prompt_setting,model,tokenizer,name):
    """
    Generate a text response by a given LLM for prompts in a list.
    
    Parameters:
    df: dataframe with questions and answers of the mgsm benchmark.
    task_lang: the language of the prompts in the dataset.
    instr_lang: the required language for the instruction prompt.
    prompt_setting: different prompting techniques: 'basic', 'cot'. 
    model: initialized model.
    tokenizer: initializer tokenizer.
    
    Returns:
    Text generated respons by the LLM for each prompt in the list.
    """
    
    pipeline = transformers.pipeline("text-generation",
                                        model=model,
                                        tokenizer=tokenizer,
                                        torch_dtype=torch.float16,
                                        device_map="auto")

    responselist = []

    for index, row in df.iterrows():

        sequences = pipeline(get_prompt(row,task,prompt_setting,instr_lang),
                                            do_sample=False, # greedy approach
                                            temperature=0.0, # t=0.0 raise error if do_sample=True
                                            repetition_penalty=1.18, # penalize the model for repeating itself
                                            num_return_sequences=1,
                                            eos_token_id=tokenizer.eos_token_id,
                                            max_new_tokens=500, # max 6 exemplars + question?
                                            return_full_text=False)
        
        for seq in sequences:
            print(get_prompt(row,task,prompt_setting,instr_lang))
            print(f"Response: {seq['generated_text']}")
            responselist.append(f"Response: {seq['generated_text']}")

    response = pd.DataFrame(data=[row.split(sep=", '") for row in responselist]) # converting the translated questionlist to a pandas df
    
    if task == 'xcopa':
        title = name + '_xcopa_' + task_lang + '_' + prompt_setting + '_instr_' + instr_lang + '.csv'

    elif task == 'mgsm':
        title = name + '_mgsm_' + task_lang + '_' + prompt_setting + '_instr_' + instr_lang + '.csv'

    elif task == 'msvamp':
        title = name + '_msvamp_' + task_lang + '_' + prompt_setting + '_instr_' + instr_lang + '.csv'

    elif task == 'coinflip':
        title = name + '_coinflip_' + task_lang + '_' + prompt_setting + '_instr_' + instr_lang + '.csv'

    elif task == 'shuffled_objects':
        title = name + '_shuffled_objects_' + task_lang + '_' + prompt_setting + '_instr_' + instr_lang + '.csv'

    response.to_csv('results/' + title, sep=';', index=False, header=False)


def calculate_accuracy(df1,df2,task):
    """
    Calculate the accuracy (% correct answers) from two input dfs.
    
    Parameters:
    df1: orginial xcopa English file with correct answer column.
    df2: response xcopa file with predicted answer column.

    Returns:
    Accuracy score (% of correct answers).
    """

    # if len(correct_answerlist) != len(predicted_answerlist):
    #     print('Unequal list length.')

    predicted_answerlist = df2['answer'].tolist()

    if task == 'mgsm' or task == 'msvamp':

        df1.iloc[:,1] = pd.to_numeric(df1.iloc[:,1])
    
        correct_answerlist = df1.iloc[:,1].tolist()

        nr_correct = sum(1 for x, y in zip(correct_answerlist, predicted_answerlist) if abs(x - y) < 1e-3)
        accuracy = round(100*(nr_correct / len(correct_answerlist)),1)

        return accuracy


    elif task == 'xcopa':
        
        correct_answerlist = df1['label'].tolist()

        map_label = {0: 'A', 1: 'B'}

        nr_correct = sum(1 for x, y in zip(correct_answerlist, predicted_answerlist) if map_label[x] == y)
        accuracy = round(100*(nr_correct / len(correct_answerlist)),1)

        return accuracy

    elif task == 'coinflip':

        correct_answerlist = df1['answer_ab'].tolist()

        nr_correct = sum(1 for x, y in zip(correct_answerlist, predicted_answerlist) if x == y)
        accuracy = round(100*(nr_correct / len(correct_answerlist)),1)

        return accuracy
    
    elif task == 'shuffled_objects':

        correct_answerlist = df1['answer_abc'].tolist()

        nr_correct = sum(1 for x, y in zip(correct_answerlist, predicted_answerlist) if x == y)
        accuracy = round(100*(nr_correct / len(correct_answerlist)),1)

        return accuracy
    
def extract_numeric_answer(inputstring):
    """
    Finds the numeric answer in the model's response.
    
    Parameters:
    inputstring: The model's response.

    Returns:
    String value of the last mentioned number.
    """
    # Regular expression to find 'the answer is ' followed by a number
    match = re.search(r'The answer is (\b\d+(?:[,.]\d+)?\b)', inputstring,re.IGNORECASE)

    if match:
        # Extract the number after 'the answer is'
        number = match.group(1)
        number = number.replace(',', '') # 
        return pd.to_numeric(number, errors='coerce')
    
    else:
        numberlist = re.findall(r'\b\d+(?:[,.]\d+)?\b',inputstring)
        
        if len(numberlist) > 0:
            number = numberlist[-1]
            if number is not None:
                number = number.replace(',', '') # 
                return pd.to_numeric(number, errors='coerce')
        else:
            return 0.0
    
def extract_abc_answer(inputstring):
    """
    Finds the multiple choice answer (A or B) in the model's response.
    
    Parameters:
    inputstring: The model's response.

    Returns:
    String value of the multiple choice answer.
    """
    matches = re.findall(r'\b[A|B|C]\b', inputstring)
    
    if len(matches) != 0:
        return matches[0]
    else: 
        return ''
    
def get_results(df,task,response_loc):

    response = pd.read_csv(response_loc,sep=';',header=None)
    response.rename(columns={0:'response'},inplace=True)
    response = response.map(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)

    answer_list = []

    if task == 'xcopa' or task == 'coinflip' or task == 'shuffled_objects':

        for i in range(len(response)):
            answer = extract_abc_answer(response.iloc[i,0])
            answer_list.append(answer)

        response['answer'] = answer_list

        return calculate_accuracy(df,response,task)
    
    elif task == 'mgsm' or task == 'msvamp':

        for i in range(len(response)):
            answer = extract_numeric_answer(response.iloc[i,0])
            answer_list.append(answer)

        response['answer'] = answer_list

        return calculate_accuracy(df,response,task)

    






######################################
############### MGSM #################
######################################

     

# def get_prompt_mgsm(question,prompt_setting,instr_lang):
#     """
#     Generate a string response by a prompt and promptsetting.
    
#     Parameters:
#     question: string task.
#     prompt_setting: different prompting techniques: 'basic', 'cot'
#     instr_lang: language of the instruction.

#     Returns:
#     String prompt.
#     """

#     # Below is an instruction that describes a task. Write a response that appropriately completes the request. 
#     # ### Instruction:
#     # {Question}
#     # ### Response: 
#     # Let's think step by step!

#     # I want you to act as a task_name expert for task_language .
#     # task_input
#     # You should retell/repeat the input_tag in English.
#     # You should task_goal .
#     # You should step-by-step answer the request.
#     # You should tell me the output_type ( output_constraint ) in this format ' output_type :'.

#     # "\nTherefore, the answer (arabic numerals) is"

#     if prompt_setting == 'cot':
#         cot = "Let's think step by step."
#     else:
#         cot = 'The answer (arabic numerals) is:'

#     # return f"Question: \n{question} \nFormulate your answer as 'The answer is [num]'. \nAnswer: {cot} "
#     return f"Question: {question} \nAnswer: {cot} "

# def mgsm_generate_response(df,task_lang,instr_lang,prompt_setting,model,tokenizer,name):
#     """
#     Generate a text response by a given LLM for prompts in a list.
    
#     Parameters:
#     df: dataframe with questions and answers of the mgsm benchmark.
#     task_lang: the language of the prompts in the dataset.
#     instr_lang: the required language for the instruction prompt.
#     prompt_setting: different prompting techniques: 'basic', 'cot'. 
#     model: initialized model.
#     tokenizer: initializer tokenizer.
    
#     Returns:
#     Text generated respons by the LLM for each prompt in the list.
#     """
#     questionlist = df.iloc[:,0].tolist()
    
#     pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     )

#     responselist = []

#     for question in questionlist:
#         sequences = pipeline(
#         get_prompt_mgsm(question,prompt_setting,instr_lang),
#         do_sample=False, # greedy approach
#         temperature=0.0, # t=0.0 raise error if do_sample=True
#         repetition_penalty=1.18, # penalize the model for repeating itself
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#         max_new_tokens=500, # max 6 exemplars + question?
#         return_full_text=False,
#         )
        
#         for seq in sequences:
#             print(get_prompt_mgsm(question,prompt_setting,instr_lang))
#             print(f"Response: {seq['generated_text']}")
#             responselist.append(f"Response: {seq['generated_text']}")
#     # print(responselist)
#     response = pd.DataFrame(data=[row.split(sep=", '") for row in responselist]) # converting the translated questionlist to a pandas df

#     title = name + '_mgsm_' + task_lang + '_' + prompt_setting + '_instr_' + instr_lang + '.csv'
#     response.to_csv('results/' + title, sep=';', index=False, header=False)
    
# def extract_numeric_answer(inputstring):
#     """
#     Finds the numeric answer in the model's response.
    
#     Parameters:
#     inputstring: The model's response.

#     Returns:
#     String value of the last mentioned number.
#     """
#     # Regular expression to find 'the answer is ' followed by a number
#     match = re.search(r'The answer is (\b\d+(?:[,.]\d+)?\b)', inputstring,re.IGNORECASE)

#     if match:
#         # Extract the number after 'the answer is'
#         number = match.group(1)
#         number = number.replace(',', '') # 
#         return pd.to_numeric(number, errors='coerce')
    
#     else:
#         numberlist = re.findall(r'\b\d+(?:[,.]\d+)?\b',inputstring)
        
#         if len(numberlist) > 0:
#             number = numberlist[-1]
#             if number is not None:
#                 number = number.replace(',', '') # 
#                 return pd.to_numeric(number, errors='coerce')
#         else:
#             return 0.0
    

# def calculate_numeric_accuracy(df1,df2):
#     """
#     Calculate the accuracy (% correct answers) from two input dfs.
    
#     Parameters:
#     df1: orginial mgsm English file with correct answer column.
#     df2: response mgsm file with predicted answer column.

#     Returns:
#     Accuracy score (% of correct answers).
#     """
#     df1.iloc[:,1] = pd.to_numeric(df1.iloc[:,1])
    
#     correct_answerlist = df1.iloc[:,1].tolist()
#     predicted_answerlist = df2['answer'].tolist()

#     if len(correct_answerlist) != len(predicted_answerlist):
#         print('Unequal list length.')

#     else:
#         # Use zip to pair up elements of the two lists and count how many pairs are equal
#         nr_correct = sum(1 for x, y in zip(correct_answerlist, predicted_answerlist) if abs(x - y) < 1e-3)
#         accuracy = round(100*(nr_correct / len(correct_answerlist)),1)

#         return accuracy
    
# def get_results(df,response_loc):

#     response = pd.read_csv(response_loc,sep=';',header=None)
#     response.rename(columns={0:'response'},inplace=True)
#     response = response.map(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)

#     answer_list = []

#     for i in range(len(response)):
#         answer = extract_numeric_answer(response.iloc[i,0])
#         answer_list.append(answer)

#     response['answer'] = answer_list

#     return calculate_numeric_accuracy(df,response)


######################################
############### XCOPA ################
######################################

# def xcopa_generate_response(df,task_lang,instr_lang,prompt_setting,model,tokenizer,name):

#     pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     )

#     responselist = []

#     if prompt_setting == 'cot':
#         cot = 'Let’s think step by step.'
#     else:
#         cot = ''
    
#     for index, row in df.iterrows():

#         # get_prompt(row,task,instr_lang_promptsetting)

#         premise = row['premise'] 
#         choice1 = row['choice1']
#         choice2 = row['choice2']
#         question = row['question']

#         prompt = f"Premise: {premise} \nOption A: {choice1} \nOption B: {choice2} \nBased on the premise, which {question} is more likely? \nPick between options A and B. \nAnswer: {cot}"

#         sequences = pipeline(
#             prompt,
#             do_sample=False, # greedy approach
#             temperature=0.0, # t=0.0 raise error if do_sample=True
#             repetition_penalty=1.18, # penalize the model for repeating itself
#             num_return_sequences=1,
#             eos_token_id=tokenizer.eos_token_id,
#             max_new_tokens=500, # max 6 exemplars + question?
#             return_full_text=False,
#             )
        
#         for seq in sequences:
#             print(prompt)
#             print(f"Response: {seq['generated_text']}")
#             responselist.append(f"Response: {seq['generated_text']}")
#         # print(responselist)
#     response = pd.DataFrame(data=[row.split(sep=", '") for row in responselist]) # converting the translated questionlist to a pandas df

#     title = name + '_xcopa_' + task_lang + '_' + prompt_setting + '_instr_' + instr_lang + '.csv'
#     response.to_csv('results/' + title, sep=';', index=False, header=False)

# def extract_abc_answer(inputstring):
#     """
#     Finds the multiple choice answer (A or B) in the model's response.
    
#     Parameters:
#     inputstring: The model's response.

#     Returns:
#     String value of the multiple choice answer.
#     """
#     matches = re.findall(r'\b[A|B|C]\b', inputstring)
    
#     if len(matches) != 0:
#         return matches[0]
#     else: 
#         return ''
    
# def calculate_abc_accuracy(df1,df2):
#     """
#     Calculate the accuracy (% correct answers) from two input dfs.
    
#     Parameters:
#     df1: orginial xcopa English file with correct answer column.
#     df2: response xcopa file with predicted answer column.

#     Returns:
#     Accuracy score (% of correct answers).
#     """
    
#     map_label = {0: 'A', 1: 'B'}

#     correct_answerlist = df1['label'].tolist()
#     predicted_answerlist = df2['answer'].tolist()

#     if len(correct_answerlist) != len(predicted_answerlist):
#         print('Unequal list length.')

#     else:
#         # Use zip to pair up elements of the two lists and count how many pairs are equal
#         nr_correct = sum(1 for x, y in zip(correct_answerlist, predicted_answerlist) if map_label[x] == y)
#         accuracy = round(100*(nr_correct / len(correct_answerlist)),1)

#         return accuracy
    
# def calculate_accuracy(df1,df2,task):
#     """
#     Calculate the accuracy (% correct answers) from two input dfs.
    
#     Parameters:
#     df1: orginial xcopa English file with correct answer column.
#     df2: response xcopa file with predicted answer column.

#     Returns:
#     Accuracy score (% of correct answers).
#     """

#     # if len(correct_answerlist) != len(predicted_answerlist):
#     #     print('Unequal list length.')

#     predicted_answerlist = df2['answer'].tolist()

#     if task == 'mgsm' or task == 'msvamp':

#         df1.iloc[:,1] = pd.to_numeric(df1.iloc[:,1])
    
#         correct_answerlist = df1.iloc[:,1].tolist()

#         nr_correct = sum(1 for x, y in zip(correct_answerlist, predicted_answerlist) if abs(x - y) < 1e-3)
#         accuracy = round(100*(nr_correct / len(correct_answerlist)),1)

#         return accuracy


#     elif task == 'xcopa':
        
#         correct_answerlist = df1['label'].tolist()

#         map_label = {0: 'A', 1: 'B'}

#         nr_correct = sum(1 for x, y in zip(correct_answerlist, predicted_answerlist) if map_label[x] == y)
#         accuracy = round(100*(nr_correct / len(correct_answerlist)),1)

#         return accuracy

#     elif task == 'coinflip':

#         correct_answerlist = df1['answer_ab'].tolist()

#         nr_correct = sum(1 for x, y in zip(correct_answerlist, predicted_answerlist) if x == y)
#         accuracy = round(100*(nr_correct / len(correct_answerlist)),1)

#         return accuracy

        
    

    

















# def msvamp_generate_response(df,task_lang,instr_lang,prompt_setting,model,tokenizer,name):
#     """
#     Generate a text response by a given LLM for prompts in a list.
    
#     Parameters:
#     df: dataframe with questions and answers of the msvamp benchmark.
#     task_lang: the language of the prompts in the dataset.
#     cot_lang: the required language for the reasoning steps.
#     prompt_setting: different prompting techniques: 'basic', 'cot'. 
#     model: initialized model.
#     tokenizer: initializer tokenizer.
#     nr_shots: number of exemplars to select.
#     shots_lang: language of the exemplars.
    
#     Returns:
#     Text generated respons by the LLM for each prompt in the list.
#     """
#     questionlist = df.iloc[:,0].tolist()
    
#     pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     )

#     responselist = []

#     for question in questionlist:
#         sequences = pipeline(
#         get_prompt_mgsm(question,prompt_setting,instr_lang),
#         do_sample=False, # greedy approach
#         temperature=0.0, # t=0.0 raise error if do_sample=True
#         repetition_penalty=1.18, # penalize the model for repeating itself
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#         max_new_tokens=500, # max 6 exemplars + question?
#         return_full_text=False,
#         )
        
#         for seq in sequences:
#             print(get_prompt_mgsm(question,prompt_setting,instr_lang))
#             print(f"Response: {seq['generated_text']}")
#             responselist.append(f"Response: {seq['generated_text']}")
#     # print(responselist)
#     response = pd.DataFrame(data=[row.split(sep=", '") for row in responselist]) # converting the translated questionlist to a pandas df

#     title = name + '_msvamp_' + task_lang + '_' + prompt_setting + '_instr_' + instr_lang + '.csv'
#     response.to_csv('results/' + title, sep=';', index=False, header=False)
#     print(title, ' saved.')


# def get_mgsm_exemplars(nr_shots,shots_lang):
#     """
#     Generate a list of n exemplar strings in the targeted language.
    
#     Parameters:
#     nr_shots: nr. of exemplars to select.
#     shots_lang: language of the exemplars.

#     Returns:
#     String with all exemplars.
#     """
#     shots_lang = language_codes[shots_lang]

#     if nr_shots > 0:
#         exemplars = pd.read_csv('datasets/mgsm/mgsm_exemplars_llama.csv', sep=';')
#         exemplars = exemplars[exemplars['language'] == shots_lang] # select target language exemplars

#         exemplar_string = ''
#         if 0 <= nr_shots <= len(exemplars):
#             sampled_exemplars = exemplars.sample(n=nr_shots, random_state=2024)
            
#             for _, row in sampled_exemplars.iterrows():
#                 exemplar_string += row.iloc[1] + ' ' + row.iloc[2] + ' '

#             return exemplar_string
        
#         else:
#             print('The nr_shots input is not correctly specified. The maximum number of exemplars is ', len(exemplars))
    
#     else:
#         return ''
#     print(title, ' saved.')