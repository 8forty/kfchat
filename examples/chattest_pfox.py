import os
import timeit

import dotenv
import openai
from dotenv import load_dotenv

import data
from llmapi import LLMAPI

load_dotenv(override=True)

env_values = dotenv.dotenv_values()


def chat(sysmsg: str, prompt: str, api: LLMAPI, model_name: str, temp: float, max_tokens: int) -> openai.ChatCompletion:
    return api.client().chat.completions.create(
        model=model_name,
        temperature=temp,  # default 1.0, 0.0->2.0
        messages=[
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,  # default 16?

        stream=False,

        # seed=27,
        # n=1,
        # top_p=1,  # default 1, ~0.01->1.0
        # frequency_penalty=1,  # default 0, -2.0->2.0
        # presence_penalty=1,  # default 0, -2.0->2.0
        # stop=[],

    )


def run(api_type_name: str, model_name: str):
    start = timeit.default_timer()
    api = LLMAPI(api_type_name, env_values)

    print(f'---- generating response from {api.type()}:{model_name}')
    response = chat(sysmsg="You are a helpful assistant that talks like Carl Sagan.",
                    prompt="How many galaxies are there?",
                    api=api,
                    model_name=model_name,
                    temp=0.7,
                    max_tokens=80)
    end = timeit.default_timer()

    print(response.choices[0].message.content)

    print(f'\n{api.type()}:{model_name} '
          f'responded with {response.usage.prompt_tokens}+{response.usage.completion_tokens} tokens '
          f'in {end - start:.0f} seconds')


models = {'openai': ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini'],
          'groq': ['llama-3.2-1b-preview', 'llama-3.3-70b-versatile', 'mixtral-8x7b-32768', 'gemma2-9b-it'],
          'azure': ['RFI-Automate-GPT-4o-mini-2000k'],
          'ollama': ['llama3.2:1b', 'llama3.2:3b', 'llama3.3:70b', 'qwen2.5:0.5b', 'gemma2:2b', 'qwq'],
          }
for atype in models.keys():
    run(atype, models[atype][0])
    print('\n')
