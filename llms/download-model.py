# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from transformers import LlamaTokenizer, LlamaForCausalLM


########### monolingual models ################

# BLOOM-7b1 
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1")

# BLOOMZ-7b1 (BLOOM finetuned on xP3 dataset)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-7b1")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-7b1")

# mT0-xxl (mT5 finetuned on xP3 dataset)
tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-xxl")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-xxl")


########### multilingual models ################


# Llama-2-7b-chat
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# BLOOMZ-7b1-mt (BLOOM finetuned on xP3mt dataset)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-7b1-mt")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-7b1-mt")

# mT5-xxl 
tokenizer = AutoTokenizer.from_pretrained("google/mt5-xxl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-xxl")

# mT0-xxl-mt (mT5 finetuned on xP3mt dataset)
tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-xxl-mt")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-xxl-mt")

# XGLM-4.5b
tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-4.5B")
model = AutoModelForCausalLM.from_pretrained("facebook/xglm-4.5B")


########### machine translation ################

#  nlln-200-3.3b
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")