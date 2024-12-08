import os
import timeit

import dotenv
import openai
from dotenv import load_dotenv

import data
from modelapi import ModelAPI

load_dotenv(override=True)

env_values = dotenv.dotenv_values()


def chat_single(sysmsg: str, prompt: str, client: openai.OpenAI, model_name: str, temp: float, max_tokens: int) -> openai.ChatCompletion:
    return client.chat.completions.create(
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


start = timeit.default_timer()
api_type_name = ['openai', 'azure', 'ollama'][2]
api_type = ModelAPI(api_type_name, env_values)
env_client = api_type.client()
env_model_name = data.apis[api_type_name]['qwq'][0]

print(f'---- generating response from {api_type.type()}:{env_model_name}')
response = chat_single(sysmsg="You are a helpful assistant that talks like Carl Sagan.",
                       prompt="How many galaxies are there?",
                       client=env_client,
                       model_name=env_model_name,
                       temp=0.7,
                       max_tokens=80)
end = timeit.default_timer()

print(response.choices[0].message.content)

print(f'\n\n----{api_type.type()}:{env_model_name} '
      f'responded with {response.usage.prompt_tokens}+{response.usage.completion_tokens} tokens '
      f'in {end - start:.0f} seconds')
