import os
import time
from time import sleep

import dotenv
import ollama
import openai
from ollama import ChatResponse
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from ollamautils import OllamaUtils

dotenv.load_dotenv(override=True)


def chat_openai_api(model_name: str, sysmsg: str):
    client = OpenAI(
        api_key=os.getenv('kfOLLAMA_API_KEY'),
        base_url=os.getenv('kfOLLAMA_OPENAI_ENDPOINT'),
    )

    print(f'model: {model_name}')
    tries = 1
    messages = [
        ChatCompletionSystemMessageParam(role='system', content=sysmsg),
        ChatCompletionUserMessageParam(role='user', content='where is paris?'),
    ]

    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                n=1,
                max_tokens=40,
                messages=messages,
            )
            print(f'response in {tries} tries: {response.choices[0].message.content}')
            # client.list_models()
            break
        except openai.RateLimitError as e:
            print('...')
            tries += 1


def chat_ollama_api(model_name: str, sysmsg: str):
    print(f'model: {model_name}')
    messages = [
        {'role': 'system', 'content': sysmsg},
        {'role': 'user', 'content': 'where is paris?'},
    ]

    tries = 0
    while True:
        try:
            tries += 1
            client = ollama.Client(host='http://localhost:11434')  # remake this every time when killing
            chat_response: ChatResponse = client.chat(
                model=model_name,
                messages=messages,
                stream=False,  # todo: allow streaming
            )
            tries_message = '' if tries == 1 else f' ({tries} tries)'
            print(f'response{tries_message}: {chat_response.message.content}')
            client.generate(model=model_name, keep_alive=0.0)  # unload the model
            OllamaUtils.kill_ollama_servers()  # todo: really!?
            break
        except ConnectionError as e:
            pass  # print(f'!!! ConnectionError on try {tries}, will retry: {e}')


def run():
    # model_names = ['llama3.2:1b', 'llama3.2:3b', 'gemma2:9b', 'gemma2:2b']
    model_names = ['llama3.2:3b']
    sysmsg = 'You are a helpful chatbot that talks in a professional manner. Your responses must always be less than 80 tokens.'

    for i in range(10):
        print(f'--- loop {i}')
        for model_name in model_names:
            # chat_openai_api(model_name, sysmsg)
            chat_ollama_api(model_name, sysmsg)


run()
