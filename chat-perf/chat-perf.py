import logging
import sys
import timeit
import traceback

import config
import message_sets
from llmconfig.llm_anthropic_config import LLMAnthropicConfig
from llmconfig.llm_openai_config import LLMOpenAIConfig, LLMOpenAISettings
from llmconfig.llmexchange import LLMMessagePair

logging.disable(logging.INFO)

all_start = timeit.default_timer()


class Data:
    llm_model_sets = {
        'base': [config.LLMData.models_by_pname['OLLAMA.llama3.2:1b']],

        'base2': [config.LLMData.models_by_pname['OLLAMA.llama3.2:1b'], config.LLMData.models_by_pname['OLLAMA.llama3.2:3b']],

        'groq-base': [
            config.LLMData.models_by_pname['GROQ.llama-3.3-70b-versatile'],
            config.LLMData.models_by_pname['GROQ.qwen-qwq-32b'],
        ],
        'groq-all': [
            config.LLMData.models_by_pname['GROQ.llama-3.3-70b-versatile'],
            config.LLMData.models_by_pname['GROQ.qwen-qwq-32b'],
            config.LLMData.models_by_pname['GROQ.gemma2-9b-it'],
            config.LLMData.models_by_pname['GROQ.deepseek-r1-distill-llama-70b'],
        ],
    }

    class LLMRawSettings(LLMOpenAISettings):
        def __init__(self, init_n: int, init_temp: float, init_top_p: float, init_max_tokens: int, init_system_message_name: str):
            """
            (almost) standard set of settings for LLMs
            todo: some LLMs/providers don't support n
            :param init_n:
            :param init_temp:
            :param init_top_p:
            :param init_max_tokens:
            :param init_system_message_name:

            """
            super().__init__(init_n=init_n, init_temp=init_temp, init_top_p=init_top_p, init_max_tokens=init_max_tokens, init_system_message_name=init_system_message_name)

    llm_settings_sets = {
        '1:800': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=800, init_system_message_name='empty'),
        ],
        'quick': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=80, init_system_message_name='empty'),
        ],
        'std4': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=800, init_system_message_name='empty'),
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=400, init_system_message_name='empty'),
            LLMRawSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=800, init_system_message_name='empty'),
            LLMRawSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=400, init_system_message_name='empty'),
        ],
        'ollama-warmup': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=800, init_system_message_name='carl-sagan'),
        ]
    }


def run(model_sets_name: str, settings_sets_name: str, message_set_name: str, csv_data: list[list[str]]):
    run_start_time = timeit.default_timer()
    model_spec: config.ModelSpec

    # llm_model_sets
    for model_spec in Data.llm_model_sets[model_sets_name]:
        settings: Data.LLMRawSettings
        print(f'{config.secs_string(all_start)}: running {model_spec.provider} {model_spec.name}...')

        # warmup the model if necessary
        if model_spec.provider == 'OLLAMA':
            warmup_start = timeit.default_timer()
            try:
                print(f'{config.secs_string(all_start)}: warmup {model_spec.provider} {model_spec.name}...')
                if model_spec.api == 'openai':
                    llm_config = LLMOpenAIConfig(model_spec.name, model_spec.provider, LLMOpenAISettings.from_settings(Data.llm_settings_sets['ollama-warmup'][0]))
                elif model_spec.api == 'anthropic':
                    llm_config = LLMAnthropicConfig(model_spec.name, model_spec.provider, LLMOpenAISettings.from_settings(Data.llm_settings_sets['ollama-warmup'][0]))
                else:
                    raise ValueError(f'api must be "openai" or "anthropic"!')
                llm_config.chat_messages([LLMMessagePair('user', 'How many galaxies are there?')])
                warmup_secs = timeit.default_timer() - warmup_start
                csv_data.append([llm_config.provider(), llm_config.model_name, '', '', '(warm-up)', '', '', str(warmup_secs)])
                print(f'{config.secs_string(all_start)}: warmup: {warmup_secs:.1f}s')
            except (Exception,) as e:
                print(f'{config.secs_string(all_start)}: warmup Exception! {model_spec.provider}:{model_spec.name}: {e.__class__.__name__}: {e} skipping...')
                traceback.print_exc(file=sys.stderr)
                break

        # llm_settings_sets
        for settings in Data.llm_settings_sets[settings_sets_name]:
            if model_spec.api == 'openai':
                llm_config = LLMOpenAIConfig(model_spec.name, model_spec.provider, LLMOpenAISettings.from_settings(settings))
            elif model_spec.api == 'anthropic':
                llm_config = LLMAnthropicConfig(model_spec.name, model_spec.provider, LLMOpenAISettings.from_settings(settings))
            else:
                raise ValueError(f'api must be "openai" or "anthropic"!')

            # llm_message_sets
            ms_start = timeit.default_timer()
            ms_input_tokens = 0
            ms_output_tokens = 0
            for idx, message_set in enumerate(message_sets.llm_message_sets[message_set_name]):
                try:
                    exchange = llm_config.chat_messages(message_set)
                except (Exception,) as e:
                    print(f'run Exception! {llm_config.provider()}:{llm_config.model_name} {message_set_name}.{message_set}: {e.__class__.__name__}: {e} skipping...')
                    break

                ms_input_tokens += exchange.input_tokens
                ms_output_tokens += exchange.output_tokens
                response_line = str(exchange.responses[0].content).replace("\n", "  ").replace('"', '""')
                print(f'{config.secs_string(all_start)}: {message_set_name}[{idx}]: '
                      f'{exchange.input_tokens}+{exchange.output_tokens} '
                      f'{timeit.default_timer() - ms_start:.1f}s  {response_line}')

            ms_end = timeit.default_timer()
            print(f'{config.secs_string(all_start)}: {message_set_name}: [{llm_config.provider()}:{llm_config.model_name}] '
                  f'{llm_config.settings().value('temp')}/{llm_config.settings().value('max_tokens')}: '
                  f'{ms_input_tokens}+{ms_output_tokens} '
                  f'{timeit.default_timer() - ms_start:.1f}s')
            csv_data.append([llm_config.provider(), llm_config.model_name, str(llm_config.settings().value('temp')), str(llm_config.settings().value('max_tokens')),
                             f'{message_set_name}',
                             str(ms_input_tokens), str(ms_output_tokens),
                             str(ms_end - ms_start)]
                            )

    run_end_time = timeit.default_timer()
    print(f'{config.secs_string(all_start)}: finished run {model_sets_name}/{settings_sets_name}/{message_set_name}: {run_end_time - run_start_time:.1f}s')


def main():
    csv_data = []

    run(model_sets_name='base', settings_sets_name='quick', message_set_name='space', csv_data=csv_data)
    # run(model_sets_name='base2', settings_sets_name='quick', message_set_name='explain', csv_data=csv_data)

    run(model_sets_name='groq-base', settings_sets_name='quick', message_set_name='space', csv_data=csv_data)

    print(f'{config.secs_string(all_start)}: finished all runs: {timeit.default_timer() - all_start:.1f}s')

    # now make CSV lines from results
    print('\n\n')
    for line in csv_data:
        print(','.join(line))


if __name__ == "__main__":
    main()
