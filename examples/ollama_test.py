import os

import dotenv
import openai
from openai import OpenAI

dotenv.load_dotenv(override=True)


def chat_openai_api(model_name: str, sysmsg: str):
    client = OpenAI(
        api_key=os.getenv('kfOLLAMA_API_KEY'),
        base_url=os.getenv('kfOLLAMA_OPENAI_ENDPOINT'),
    )

    print(f'model: {model_name}')
    tries = 1
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                n=1,
                max_tokens=40,
                messages=[
                    {'role': 'system', 'content': sysmsg},
                    {'role': 'user', 'content': 'where is paris?'}
                ]
            )
            print(f'response in {tries} tries: {response.choices[0].message.content}')
            # client.list_models()
            break
        except openai.RateLimitError as e:
            print('...')
            tries += 1


if __name__ == "__main__":
    # model_names = ['llama3.2:1b', 'llama3.2:3b', 'gemma2:9b', 'gemma2:2b']
    model_names = ['llama3.2:1b', 'llama3.2:3b']
    pro80 = 'You are a helpful chatbot that talks in a professional manner. Your responses must always be less than 80 tokens.'

    for model_name in model_names:
        chat_openai_api(model_name, pro80)
