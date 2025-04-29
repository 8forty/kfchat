import logging
import sys
import timeit
import traceback

import config
import prompt_sets
from llmconfig.llm_anthropic_config import LLMAnthropicConfig
from llmconfig.llm_openai_config import LLMOpenAIConfig, LLMOpenAISettings
from llmconfig.llmexchange import LLMMessagePair

logging.disable(logging.INFO)

all_start = timeit.default_timer()


class Data:
    llm_model_sets = {
        'base': [
            config.LLMData.models_by_pname['OLLAMA.llama3.2:1b'],
            config.LLMData.models_by_pname['OLLAMA.llama3.2:3b'],
        ],

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

        'gorbash-test': [
            config.LLMData.models_by_pname['OLLAMA.llama3.2:3b'],
            config.LLMData.models_by_pname['OLLAMA.mistral-nemo:12b'],
            config.LLMData.models_by_pname['OLLAMA.mixtral:8x7b'],
            config.LLMData.models_by_pname['OLLAMA.gemma2:9b-instruct-fp16'],
            config.LLMData.models_by_pname['OLLAMA.gemma2:9b-text-fp16'],
            config.LLMData.models_by_pname['OLLAMA.gemma3:1b'],
            config.LLMData.models_by_pname['OLLAMA.gemma3:4b'],
            config.LLMData.models_by_pname['OLLAMA.gemma3:12b'],
            config.LLMData.models_by_pname['OLLAMA.gemma3:12b-it-fp16'],
            config.LLMData.models_by_pname['OLLAMA.gemma3:27b'],
            config.LLMData.models_by_pname['OLLAMA.gemma3:27b-it-fp16'],
            config.LLMData.models_by_pname['OLLAMA.llama3.3:70b'],
            config.LLMData.models_by_pname['OLLAMA.llama3.3:70b-instruct-q2_K'],
            config.LLMData.models_by_pname['OLLAMA.deepseek-r1:32b'],
            config.LLMData.models_by_pname['OLLAMA.deepseek-v2:16b'],
            config.LLMData.models_by_pname['OLLAMA.qwq:latest'],
            config.LLMData.models_by_pname['OLLAMA.phi4:14b'],
            config.LLMData.models_by_pname['OLLAMA.phi4:14b-q8_0'],
            config.LLMData.models_by_pname['OLLAMA.phi4:14b-fp16'],
            config.LLMData.models_by_pname['OLLAMA.olmo2:13b'],
            config.LLMData.models_by_pname['OLLAMA.command-r7b'],
            config.LLMData.models_by_pname['OLLAMA.openthinker:32b'],
            config.LLMData.models_by_pname['OLLAMA.qwen3:14b-q8_0'],
            config.LLMData.models_by_pname['OLLAMA.qwen3:30b-a3b'],
            config.LLMData.models_by_pname['OLLAMA.qwen3:30b-a3b-q4_K_M'],
            config.LLMData.models_by_pname['OLLAMA.qwen3:32b-q4_K_M'],
            config.LLMData.models_by_pname['OLLAMA.qwen3:32b'],
        ],
    }

    class LLMRawSettings(LLMOpenAISettings):
        def __init__(self, init_n: int, init_temp: float, init_top_p: float, init_max_tokens: int,
                     init_system_message_name: str):
            """
            (almost) standard set of settings for LLMs
            todo: some LLMs/providers don't support n
            :param init_n:
            :param init_temp:
            :param init_top_p:
            :param init_max_tokens:
            :param init_system_message_name:

            """
            super().__init__(init_n=init_n, init_temp=init_temp, init_top_p=init_top_p,
                             init_max_tokens=init_max_tokens, init_system_message_name=init_system_message_name)

    #         'convo': conversational_sysmsg,
    #         'convo80': conversational80_sysmsg,
    #         'professional': professional_sysmsg,
    #         'professional80': professional80_sysmsg,
    #         'professional800': professional800_sysmsg,
    #         'technical': technical_sysmsg,
    #         'technical80': technical80_sysmsg,
    #         'technical800': technical800_sysmsg,
    #         'text-sentiment': textclass_sysmsg,
    #         'carl-sagan': csagan_sysmsg,
    #         'empty': empty_sysmsg,
    llm_settings_sets = {
        '1:800': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=800,
                           init_system_message_name='empty'),
        ],
        'quick': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=80,
                           init_system_message_name='empty'),
        ],
        'std4': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=800,
                           init_system_message_name='empty'),
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=400,
                           init_system_message_name='empty'),
            LLMRawSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=800,
                           init_system_message_name='empty'),
            LLMRawSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=400,
                           init_system_message_name='empty'),
        ],
        'gorbash-test': [
            LLMRawSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=800,
                           init_system_message_name='professional800'),
        ],
        'ollama-warmup': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=800,
                           init_system_message_name='carl-sagan'),
        ]
    }


def run(model_sets_name: str, settings_set_name: str, prompt_set_name: str, csv_data: list[list[str]]):
    run_start_time = timeit.default_timer()
    model_spec: config.ModelSpec
    csv_data.append(['provider', 'model', 'temp', 'max_tokens', 'sysmsg', 'prompt-set', 'tokens-in', 'tokens-out',
                     'seconds', 'last-response-1line'])

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
                    llm_config = LLMOpenAIConfig(model_spec.name, model_spec.provider,
                                                 LLMOpenAISettings.from_settings(Data.llm_settings_sets['ollama-warmup'][0]))
                elif model_spec.api == 'anthropic':
                    llm_config = LLMAnthropicConfig(model_spec.name, model_spec.provider,
                                                    LLMOpenAISettings.from_settings(Data.llm_settings_sets['ollama-warmup'][0]))
                else:
                    raise ValueError(f'api must be "openai" or "anthropic"!')
                llm_config.chat_messages(messages=[LLMMessagePair('user', 'How many galaxies are there?')])
                warmup_secs = timeit.default_timer() - warmup_start
                csv_data.append([llm_config.provider(), llm_config.model_name, '', '', '', '(warm-up)', '', '', f'{warmup_secs:.1f}', ''])
                print(f'{config.secs_string(all_start)}: warmup: {warmup_secs:.1f}s')
            except (Exception,) as e:
                print(f'{config.secs_string(all_start)}: warmup Exception! {model_spec.provider}:{model_spec.name}: {e.__class__.__name__}: {e} skipping...')
                traceback.print_exc(file=sys.stderr)
                break

        # llm_settings_sets
        response_line: str = ''
        for settings in Data.llm_settings_sets[settings_set_name]:
            if model_spec.api == 'openai':
                llm_config = LLMOpenAIConfig(model_spec.name, model_spec.provider, LLMOpenAISettings.from_settings(settings))
            elif model_spec.api == 'anthropic':
                llm_config = LLMAnthropicConfig(model_spec.name, model_spec.provider, LLMOpenAISettings.from_settings(settings))
            else:
                raise ValueError(f'api must be "openai" or "anthropic"!')

            ms_start = timeit.default_timer()
            ms_input_tokens = 0
            ms_output_tokens = 0
            # prompts loop
            for idx, prompt_set in enumerate(prompt_sets.llm_prompt_sets[prompt_set_name]):
                try:
                    exchange = llm_config.chat_messages(prompt_set)
                except (Exception,) as e:
                    print(f'run Exception! {llm_config.provider()}:{llm_config.model_name} '
                          f'{prompt_set_name}.{prompt_set}: {e.__class__.__name__}: {e} skipping...')
                    break

                ms_input_tokens += exchange.input_tokens
                ms_output_tokens += exchange.output_tokens
                response_line = (str(exchange.responses[0].content).replace("\n", "  ")
                                 .replace('"', '""'))
                print(f'{config.secs_string(all_start)}: {prompt_set_name}[{idx}]: '
                      f'{exchange.input_tokens}->{exchange.output_tokens} '
                      f'{timeit.default_timer() - ms_start:.1f}s  {response_line}')

            ms_end = timeit.default_timer()
            print(f'{config.secs_string(all_start)}: {prompt_set_name}: '
                  f'[{llm_config.provider()}:{llm_config.model_name}] '
                  f'{llm_config.settings().value('temp')}/{llm_config.settings().value('max_tokens')} '
                  f'{llm_config.settings().value('system_message_name')}: '
                  f'{ms_input_tokens}+{ms_output_tokens} '
                  f'{timeit.default_timer() - ms_start:.1f}s')

            # csv
            csv_data.append([llm_config.provider(), llm_config.model_name,
                             str(llm_config.settings().value('temp')),
                             str(llm_config.settings().value('max_tokens')),
                             str(llm_config.settings().value('system_message_name')),
                             f'{prompt_set_name}',
                             str(ms_input_tokens), str(ms_output_tokens),
                             f'{ms_end - ms_start:.1f}',
                             f'"{response_line}"']
                            )

    run_end_time = timeit.default_timer()
    print(f'{config.secs_string(all_start)}: finished run {model_sets_name}/{settings_set_name}/{prompt_set_name}: '
          f'{run_end_time - run_start_time:.1f}s')


def main():
    csv_data = []

    # run(model_sets_name='base', settings_sets_name='quick', prompt_set_name='space', csv_data=csv_data)
    run(model_sets_name='gorbash-test', settings_set_name='gorbash-test', prompt_set_name='gorbash-test', csv_data=csv_data)
    # run(model_sets_name='groq-base', settings_sets_name='quick', prompt_set_name='space', csv_data=csv_data)

    print(f'{config.secs_string(all_start)}: finished all runs: {timeit.default_timer() - all_start:.1f}s')

    # now make CSV lines from results
    print('\n\n')
    for line in csv_data:
        print(','.join(line))


if __name__ == "__main__":
    main()
