import timeit

import dotenv
from dotenv import load_dotenv

from llmoaiconfig import LLMOaiConfig, LLMOaiExchange

load_dotenv(override=True)

env_values = dotenv.dotenv_values()


def chat(sysmsg: str, prompt: str, cfg: LLMOaiConfig, model_name: str, temp: float, max_tokens: int) -> LLMOaiExchange:
    return cfg.run_chat_completion(
        model_name=model_name,
        temp=temp,  # default 1.0, 0.0->2.0
        top_p=1.0,  # default 1, ~0.01->1.0
        max_tokens=max_tokens,  # default 16?
        n=1,
        convo=[
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": prompt},
        ],

        # stream=False,
        # seed=27,
        # frequency_penalty=1,  # default 0, -2.0->2.0
        # presence_penalty=1,  # default 0, -2.0->2.0
        # stop=[],

    )


def run(api_type_name: str, model_name: str):
    start = timeit.default_timer()
    cfg = LLMOaiConfig('chattest', api_type_name, env_values, model_name, init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=80,
                       init_system_message="You are a helpful assistant that talks like Carl Sagan.")

    print(f'---- generating response from {cfg.api_type()}:{model_name}')
    exchange = chat(sysmsg=cfg.system_message,
                    prompt="How many galaxies are there?",
                    cfg=cfg,
                    model_name=cfg.model_name,
                    temp=cfg.temp,
                    max_tokens=cfg.max_tokens)
    end = timeit.default_timer()

    print(exchange.completion.choices[0].message.content)

    print(f'\n{cfg.api_type()}:{model_name} '
          f'responded with {exchange.completion.usage.prompt_tokens}+{exchange.completion.usage.completion_tokens} tokens '
          f'in {end - start:.0f} seconds')


models = {'openai': ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini'],
          'groq': ['llama-3.2-1b-preview', 'llama-3.3-70b-versatile', 'mixtral-8x7b-32768', 'gemma2-9b-it'],
          'azure': ['RFI-Automate-GPT-4o-mini-2000k'],
          'ollama': ['llama3.2:1b', 'llama3.2:3b', 'llama3.3:70b', 'qwen2.5:0.5b', 'gemma2:2b', 'qwq'],
          }
for atype in models.keys():
    run(atype, models[atype][0])
    print('\n')
