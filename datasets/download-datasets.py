from datasets import load_dataset

# XCOPA: ML/CL commonsense reasoning
# XStoryCloze: commonsense reasoning
# MGSM: arithmetic reasoning
# MKQA: question answering
# XNLI: natural language inference
# PAWS-X: Paraphrase identification
# XLSUM: summarization
# FLORES: machine translation

dataset = load_dataset("juletxara/xcopa_mt")
dataset = load_dataset("juletxara/xstory_cloze_mt")
dataset = load_dataset("juletxara/mgsm_mt")
dataset = load_dataset("mkqa")
dataset = load_dataset("juletxara/pawsx_mt")
dataset = load_dataset("juletxara/xnli_mt")
dataset = load_dataset("csebuetnlp/xlsum")
dataset = load_dataset("facebook/flores")