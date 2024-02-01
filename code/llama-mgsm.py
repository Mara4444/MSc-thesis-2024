from src.cot_utils import *
from src.dataset_utils import *

######################### todo #############################
# - decide on prompt_setting
# - how to work with few shot examples?
# - create function that extracts the answer from the response
# - create function that calculates the accuracy
# - create function get dataset for translated datasets
# - create function that <after mgsm_generate_response> makes sure that the response df has 1 response string per row and extracts the answer for each response and puts this as new column in df
# - run the function for each mgsm task_lang + cot_lang combination

# inputs
dataset = get_dataset("mgsm","fr")

df = pd.DataFrame(data={'question' : dataset["test"]["question"],
                        'answer_number' : dataset["test"]["answer_number"]})
df = df[:4]

prompt_setting = "basic"
task_lang = "French"
cot_lang = "English"

mgsm_generate_response(df,task_lang,cot_lang,prompt_setting,model,tokenizer)