from src.cot_utils import *
from src.dataset_utils import *

# Llama-2 model

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

English = get_dataset_df("coinflip","eng_Latn")

generate_response(df = English,
                  task = "coinflip",
                  task_lang = "English",
                  instr_lang = "English",
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "llama-2-7b")
    
generate_response(df = English,
                  task = "coinflip",
                  task_lang = "English",
                  instr_lang = "English",
                  prompt_setting = "cot",
                  model = model,
                  tokenizer = tokenizer,
                  name = "llama-2-7b")