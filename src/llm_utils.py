import os
import requests
from dotenv import load_dotenv

class CLGPT:
    '''
    Main class for CL generation using OpenAI GPT.
    '''
    def __init__(self, model='gpt-3.5-turbo', **kwargs):
        '''
        Args:
            model: str: gpt model version
            kwargs: dict: additional arguments for CL generation
        '''
        self.model = model
        self.kwargs = kwargs

    def gen_cl(self, prompt, tgt_lang, **kwargs):
        '''
        Generate a response given an input prompt.
        Args:
            prompt: str: input prompt
            tgt_lang: str: target language
            kwargs: dict: additional arguments for CL generation
        '''
        load_dotenv()
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        if OPENAI_API_KEY is None:
            raise Exception("Please set your OPENAI_API_KEY environment variable.")
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-type": "application/json"
        }

        api_url = f'https://api.openai.com/v1/chat/completions' # API endpoint

        # create data payload
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a cross-lingual chatbot. Given the following prompt, generate a response in {tgt_lang}."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.7),
            "n": kwargs.get("n", 1),
        }

        # merge additional kwargs
        data.update(self.kwargs)

        # call the API with the prompt
        response = requests.post(api_url, headers=headers, json=data)

        if response.status_code == 200: # success
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
        else:
            # print error message
            raise Exception(f"Request failed with code {response.status_code}, {response.json().get('error', 'Unknown error')}")
    
    