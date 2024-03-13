from src.cot_utils import *
from src.dataset_utils import *

# Bloomz model

model_name = "bigscience/bloomz-7b1-mt"
model = BloomForCausalLM.from_pretrained(model_name)
tokenizer = BloomTokenizerFast.from_pretrained(model_name)

# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("cmarkea/bloomz-7b1-mt-sft-chat")
# model = AutoModelForCausalLM.from_pretrained("cmarkea/bloomz-7b1-mt-sft-chat")

English = get_dataset_df("mgsm","en")

mgsm_generate_response(df=English,
                        task_lang="English",
                        instr_lang="English",
                        prompt_setting="basic",
                        model=model,
                        tokenizer=tokenizer,
                        name="bloomz-7b1")

mgsm_generate_response(df=English,
                        task_lang="English",
                        instr_lang="English",
                        prompt_setting="cot",
                        model=model,
                        tokenizer=tokenizer,
                        name="bloomz-7b1")
