from datasets import load_dataset

# XCOPA: commonsense reasoning
# XStoryCloze: commonsense reasoning
# MGSM: arithmetic reasoning
# MKQA: question answering
# XNLI: natural language inference
# PAWS-X: Paraphrase identification
# XLSUM: summarization
# FLORES: machine translation

# dataset = load_dataset("juletxara/mgsm","en")           # ["en","bn","de","es","fr","ja","ru","sw","te","th","zh"]
# dataset = load_dataset("pkavumba/balanced-copa")        # English only
# dataset = load_dataset("xcopa","et")                    # ["et","ht","id","it","qu","sw","ta","th","tr"]
# dataset = load_dataset("juletxara/xstory_cloze","en")   # ['ar', 'en', 'es', 'eu', 'hi', 'id', 'my', 'ru', 'sw', 'te', 'zh']
# dataset = load_dataset("mkqa")                          # ['ar', 'da', 'de', 'en', 'es', 'fi', 'fr', 'he', 'hu', 'it','ja', 'km', 'ko', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sv', 'th', 'tr', 'vi', 'zh']
# dataset = load_dataset("paws-x","en")                   # ['de', 'en', 'es', 'fr', 'ja', 'ko', 'zh']
# dataset = load_dataset("xnli","en")                     # ['all_languages', 'ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
# dataset = load_dataset("csebuetnlp/xlsum","english")    # ['amharic', 'arabic', 'azerbaijani', 'bengali', 'burmese', 'chinese_simplified', 'chinese_traditional', 'english', 'french', 'gujarati', 'hausa', 'hindi', 'igbo', 'indonesian', 'japanese', 'kirundi', 'korean', 'kyrgyz', 'marathi', 'nepali', 'oromo', 'pashto', 'persian', 'pidgin', 'portuguese', 'punjabi', 'russian', 'scottish_gaelic', 'serbian_cyrillic', 'serbian_latin', 'sinhala', 'somali', 'spanish', 'swahili', 'tamil', 'telugu', 'thai', 'tigrinya', 'turkish', 'ukrainian', 'urdu', 'uzbek', 'vietnamese', 'welsh', 'yoruba']
# dataset = load_dataset("facebook/flores","eng_Latn")    # example "eng_Latn"

def get_dataset(name,lang):
    """
    Loads a dataset from huggingface in the requested language.
    
    Parameters:
    name: name of the dataset ['mgsm', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum']
    lang: language of the dataset to load.
    
    Returns:
    Dataset in the specified language.
    """
    if name == "mgsm":
        dataset = load_dataset("juletxara/mgsm",lang) 
        
        return dataset
    
    elif name == "xcopa" and lang == "en":
        dataset = load_dataset("pkavumba/balanced-copa")
        
        return dataset
    
    elif name == "xcopa":
        dataset = load_dataset("xcopa",lang) 
        
        return dataset 
    
    elif name == "xstorycloze":
        dataset = load_dataset("juletxara/xstory_cloze",lang)   
        
        return dataset
    
    elif name == "mkqa":
        dataset = load_dataset("mkqa")
        
        if lang in dataset["train"]["queries"][0].keys():
            questionlist = [language[lang] for language in dataset["train"]["queries"]]
            answerlist = [language[lang] for language in dataset["train"]["answers"]]
            
            dataset = {
                "train": {
                    "queries": questionlist,
                    "answers": answerlist,
                }
            }

            return dataset
        
        else:
            print("Language not found. Specify one of the following languages: ['ar', 'da', 'de', 'en', 'es', 'fi', 'fr', 'he', 'hu', 'it','ja', 'km', 'ko', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sv', 'th', 'tr', 'vi', 'zh'")
    
    elif name == "pawsx":
        dataset = load_dataset("paws-x",lang)    

        return dataset
    
    elif name == "xnli":
        dataset = load_dataset("xnli",lang)  

        return dataset
    
    elif name == "xlsum":        
        dataset = load_dataset("csebuetnlp/xlsum",lang) # ['amharic', 'arabic', 'azerbaijani', 'bengali', 'burmese', 'chinese_simplified', 'chinese_traditional', 'english', 'french', 'gujarati', 'hausa', 'hindi', 'igbo', 'indonesian', 'japanese', 'kirundi', 'korean', 'kyrgyz', 'marathi', 'nepali', 'oromo', 'pashto', 'persian', 'pidgin', 'portuguese', 'punjabi', 'russian', 'scottish_gaelic', 'serbian_cyrillic', 'serbian_latin', 'sinhala', 'somali', 'spanish', 'swahili', 'tamil', 'telugu', 'thai', 'tigrinya', 'turkish', 'ukrainian', 'urdu', 'uzbek', 'vietnamese', 'welsh', 'yoruba']

        return dataset
    
    else:
        print("Dataset name is not correctly specified. Please input 'mgsm', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum'.")

######### state which dataset to download ########
get_dataset("mgsm","en")