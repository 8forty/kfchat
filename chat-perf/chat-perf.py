import logging
import subprocess
import sys
import timeit
import traceback

import ollamautils

import config
from cpfunctions import CPFunctions
from cpdata import CPData, CPRunType, CPRunSpec
from llmconfig.llm_anthropic_config import LLMAnthropicConfig
from llmconfig.llm_openai_config import LLMOpenAIConfig, LLMOpenAISettings
from llmconfig.llmexchange import LLMMessagePair, LLMExchange
from ollamautils import OllamaUtils

logging.disable(logging.INFO)

all_start = timeit.default_timer()


def ollama_ps(model: config.ModelSpec) -> str:
    if model.provider == 'OLLAMA':
        ollama_out = subprocess.run(['ollama', 'ps'], capture_output=True, text=True).stdout
        o = ollama_out.splitlines()[1].split()
        return f'{o[0]},{o[2]}{o[3]},{o[4]},{o[5]}'
    else:
        return ''


def run(run_set_name: str, settings_set_name: str, prompt_set_name: str, csv_data: list[list[str]]):
    run_start_time = timeit.default_timer()
    run_set: CPRunSpec
    csv_data.append(['provider', 'model', 'temp', 'max_tokens', 'sysmsg', 'prompt-set', 'tokens-in', 'tokens-out',
                     'seconds', 'ollama', 'last-response-1line'])

    # llm_model_sets
    for run_set in CPData.run_sets[run_set_name]:
        print(f'{config.secs_string(all_start)}: running {run_set.run_type} {run_set.model.provider} {run_set.model.name}...')

        if run_set.run_type in [CPRunType.LLM, CPRunType.RAG]:
            model = run_set.model
            settings: CPData.LLMRawSettings

            # warmup the model if necessary
            if model.provider == 'OLLAMA':
                warmup_start = timeit.default_timer()

                ul_response = OllamaUtils.unload_all()
                print(f'{config.secs_string(all_start)}: warmup: unloaded models {ul_response}')

                try:
                    print(f'{config.secs_string(all_start)}: warmup {model.provider} {model.name}...')
                    if model.api == 'openai':
                        llm_config = LLMOpenAIConfig(model.name, model.provider,
                                                     LLMOpenAISettings.from_settings(CPData.llm_settings_sets['ollama-warmup'][0]))
                    elif model.api == 'anthropic':
                        llm_config = LLMAnthropicConfig(model.name, model.provider,
                                                        LLMOpenAISettings.from_settings(CPData.llm_settings_sets['ollama-warmup'][0]))
                    else:
                        raise ValueError(f'api must be "openai" or "anthropic"!')

                    while True:
                        # run the llm
                        llm_config.chat_messages(messages=[LLMMessagePair('user', 'How many galaxies are there?')])

                        # check that the correct model is running
                        if not OllamaUtils.is_model_running(model.name):
                            print(f'{config.secs_string(all_start)}: warmup: !! model {model.name} isn''t running!  retrying warmup...')
                        else:
                            break

                    warmup_secs = timeit.default_timer() - warmup_start
                    csv_data.append([llm_config.provider(), llm_config.model_name, '', '', '', '(warm-up)', '', '',
                                     f'{warmup_secs:.1f}', f'"{ollama_ps(model)}"', ''])
                    print(f'{config.secs_string(all_start)}: warmup: {warmup_secs:.1f}s')
                except (Exception,) as e:
                    print(f'{config.secs_string(all_start)}: warmup Exception! {model.provider}:{model.name}: {e.__class__.__name__}: {e} skipping...')
                    traceback.print_exc(file=sys.stderr)
                    break

            # settings loop
            response_line: str = ''
            for settings in CPData.llm_settings_sets[settings_set_name]:
                if model.api == 'openai':
                    llm_config = LLMOpenAIConfig(model.name, model.provider, LLMOpenAISettings.from_settings(settings))
                elif model.api == 'anthropic':
                    llm_config = LLMAnthropicConfig(model.name, model.provider, LLMOpenAISettings.from_settings(settings))
                else:
                    raise ValueError(f'api must be "openai" or "anthropic"!')

                ls = timeit.default_timer()
                ls_input_tokens = 0
                ls_output_tokens = 0
                # prompts loop
                for idx, prompt_set in enumerate(CPData.llm_prompt_sets[prompt_set_name]):
                    exchange: LLMExchange | None = None
                    try:
                        if run_set.run_type == CPRunType.LLM:
                            exchange = CPFunctions.run_llm_prompt(prompt_set, None, llm_config, all_start)
                        elif run_set.run_type == CPRunType.RAG:
                            # todo: configure
                            exchange = CPFunctions.run_rag_prompt(prompt_set, run_set.collection_name, llm_config,
                                                                  0, 0.5, all_start)
                        if exchange is None:
                            raise ValueError(f'exchange is None!')
                    except (Exception,) as e:
                        print(f'run Exception! {llm_config.provider()}:{llm_config.model_name} '
                              f'{prompt_set_name}.{prompt_set}: {e.__class__.__name__}: {e} skipping...')
                        break

                    ls_input_tokens += exchange.input_tokens
                    ls_output_tokens += exchange.output_tokens
                    response_line = (str(exchange.responses[0].content).replace("\n", "  ")
                                     .replace('"', '""'))
                    print(f'{config.secs_string(all_start)}: {prompt_set_name}[{idx}]: '
                          f'{exchange.input_tokens}->{exchange.output_tokens} '
                          f'{timeit.default_timer() - ls:.1f}s  {response_line}')

                ms_end = timeit.default_timer()
                print(f'{config.secs_string(all_start)}: {prompt_set_name}: '
                      f'[{llm_config.provider()}:{llm_config.model_name}] '
                      f'{llm_config.settings().value('temp')}/{llm_config.settings().value('max_tokens')} '
                      f'{llm_config.settings().value('system_message_name')}: '
                      f'{ls_input_tokens}+{ls_output_tokens} '
                      f'{timeit.default_timer() - ls:.1f}s')
                print(f'{config.secs_string(all_start)}: ollama ps: {ollama_ps(model)}')

                # csv
                csv_data.append([llm_config.provider(), llm_config.model_name,
                                 str(llm_config.settings().value('temp')),
                                 str(llm_config.settings().value('max_tokens')),
                                 str(llm_config.settings().value('system_message_name')),
                                 f'{prompt_set_name}',
                                 str(ls_input_tokens), str(ls_output_tokens),
                                 f'{ms_end - ls:.1f}',
                                 f'"{ollama_ps(model)}"',
                                 f'"{response_line}"']
                                )

    run_end_time = timeit.default_timer()
    print(f'{config.secs_string(all_start)}: finished run {run_set_name}/{settings_set_name}/{prompt_set_name}: '
          f'{run_end_time - run_start_time:.1f}s')


def main():
    csv_data = []

    # run(run_set_name='base', settings_set_name='quick', prompt_set_name='space', csv_data=csv_data)
    run(run_set_name='gorbash-test-kf', settings_set_name='gorbash-test', prompt_set_name='gorbash-test', csv_data=csv_data)

    print(f'{config.secs_string(all_start)}: finished all runs: {timeit.default_timer() - all_start:.1f}s')

    # now make CSV lines from results
    print('\n\n')
    for line in csv_data:
        print(','.join(line))


if __name__ == "__main__":
    # pydevd.settrace('localhost', port=5678, stdoutToServer=True, stderrToServer=True)
    main()
