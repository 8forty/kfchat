import dotenv
from dotenv import load_dotenv

from llmconfig.llm_anthropic_config import LLMAnthropicSettings
from llmconfig.llmconfig import LLMConfig
from llmconfig.llm_openai_config import LLMOpenAISettings, LLMOpenAIConfig

load_dotenv(override=True)

env_values = dotenv.dotenv_values()


def run(cfg: LLMConfig):
    print(f'---- generating response from {cfg.provider()}:{cfg.model_name}')
    exchange = cfg.chat_convo(convo=[],
                              prompt="How many galaxies are there?")

    print(exchange.responses[0].content)

    print(f'\n{cfg.provider()}:{cfg.model_name} '
          f'responded with {exchange.input_tokens}+{exchange.output_tokens} tokens '
          f'in {exchange.response_duration_secs:.0f} seconds')


settings_openai = LLMOpenAISettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=80, init_system_message_name='carl-sagan')
settings_anthropic = LLMAnthropicSettings(init_temp=0.7, init_top_p=1.0, init_max_tokens=80, init_system_message_name='carl-sagan')
models = {
    # 'openai': ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini'],
    # 'groq': ['llama-3.2-1b-preview', 'llama-3.3-70b-versatile', 'mixtral-8x7b-32768', 'gemma2-9b-it'],
    # 'azure': ['RFI-Automate-GPT-4o-mini-2000k'],
    # 'ollama': ['llama3.2:1b', 'llama3.2:3b', 'llama3.3:70b', 'qwen2.5:0.5b', 'gemma2:2b', 'qwq'],
    'x': [
        LLMOpenAIConfig(model_name='llama3.2:1b', provider='OLLAMA', settings=settings_openai),
    ]
}
for k in models.keys():
    run(models[k][0])
    print('\n')
