import timeit

import dotenv
from dotenv import load_dotenv

from llmconfig.llmoaiconfig import LLMOaiConfig, LLMOaiExchange, LLMOaiSettings

load_dotenv(override=True)

env_values = dotenv.dotenv_values()


def chat(sysmsg: str, prompt: str, cfg: LLMOaiConfig) -> LLMOaiExchange:
    return cfg.chat_messages([{"role": "system", "content": sysmsg}, {"role": "user", "content": prompt}])


def run(provider_name: str, model_name: str):
    start = timeit.default_timer()
    cfg = LLMOaiConfig(model_name, provider_name,
                       LLMOaiSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=80, init_system_message_name="carl-sagan"))

    print(f'---- generating response from {cfg.provider()}:{model_name}')
    exchange = chat(sysmsg=cfg.settings.system_message,
                    prompt="How many galaxies are there?",
                    cfg=cfg)
    end = timeit.default_timer()

    print(exchange.responses[0].content)

    print(f'\n{cfg.provider()}:{model_name} '
          f'responded with {exchange.input_tokens}+{exchange.output_tokens} tokens '
          f'in {end - start:.0f} seconds')


models = {
    # 'openai': ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini'],
    # 'groq': ['llama-3.2-1b-preview', 'llama-3.3-70b-versatile', 'mixtral-8x7b-32768', 'gemma2-9b-it'],
    # 'azure': ['RFI-Automate-GPT-4o-mini-2000k'],
    'ollama': ['llama3.2:1b', 'llama3.2:3b', 'llama3.3:70b', 'qwen2.5:0.5b', 'gemma2:2b', 'qwq'],
}
for ptype in models.keys():
    run(ptype, models[ptype][0])
    print('\n')
