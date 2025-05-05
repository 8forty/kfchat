import asyncio
import logging
import sys
import time
import timeit
import traceback

import ollama

import config
from cpdata import CPData, CPRunType, CPRunSpec
from cpfunctions import CPFunctions
from llmconfig.llm_anthropic_config import LLMAnthropicConfig, LLMAnthropicSettings
from llmconfig.llm_ollama_config import LLMOllamaConfig, LLMOllamaSettings
from llmconfig.llm_openai_config import LLMOpenAIConfig, LLMOpenAISettings
from llmconfig.llmexchange import LLMExchange
from ollamautils import OllamaUtils

logging.disable(logging.INFO)

all_start = timeit.default_timer()


def ollama_ps(model: config.ModelSpec) -> str:
    if model.provider == 'OLLAMA':
        # ProcessResponse(models=[Model(model='gemma3:1b', name='gemma3:1b',
        # digest='8648f39daa8fbf5b18c7b4e6a8fb4990c692751d49917417b8842ca5758e7ffc',
        # expires_at=datetime.datetime(2025, 5, 4, 11, 5, 25, 506872, tzinfo=TzInfo(-07:00)), size=1907176448, size_vram=1907176448,
        # details=ModelDetails(parent_model='', format='gguf', family='gemma3', families=['gemma3'], parameter_size='999.89M',
        # quantization_level='Q4_K_M'))])
        retval = ''
        for m in ollama.ps().models:
            gpu = float(m.size) / float(m.size_vram) if m.size_vram > 0 else 0.0
            cpu = 1.0 - gpu
            load = f'{cpu * 100.0:.0f}%/{gpu * 100.0:.0f}%,CPU/GPU' if cpu > 0.0 else f'100%,GPU'
            retval += f'{m.name},{m.details.quantization_level},{float(m.size) / (1024.0 * 1024.0 * 1024.0):.1f}GB,{load} '

        return retval
    else:
        return ''


def run(run_set_name: str, settings_set_name: str, sysmsg_name: str, prompt_set_name: str, csv_data: list[list[str]]):
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
                warmup_retries = 0
                warmup_retry_wait_secs = 1.0
                warmup_start = timeit.default_timer()
                try:
                    while True:
                        print(f'{config.secs_string(all_start)}: warmup {model.provider} {model.name}...')
                        # todo: factory this shit
                        if model.api.upper() == 'OPENAI':
                            llm_config = LLMOpenAIConfig(model.name, model.provider,
                                                         LLMOpenAISettings.from_settings(CPData.llm_settings_sets['ollama-warmup'][0]))
                        elif model.api.upper() == 'ANTHROPIC':
                            llm_config = LLMAnthropicConfig(model.name, model.provider,
                                                            LLMAnthropicSettings.from_settings(CPData.llm_settings_sets['ollama-warmup'][0]))
                        elif model.api.upper() == 'OLLAMA':
                            llm_config = LLMOllamaConfig(model.name, model.provider,
                                                         LLMOllamaSettings.from_settings(CPData.llm_settings_sets['ollama-warmup'][0]))
                        else:
                            raise ValueError(f'api must be "openai" or "anthropic" or "ollama"!')

                        while True:
                            # run the llm
                            CPFunctions.run_llm_prompt(CPData.llm_prompt_sets['galaxies'][0], None, llm_config, all_start)

                            # check that the correct model is running
                            if not OllamaUtils.is_model_running(model.name):
                                running = [m.name for m in ollama.ps().models]
                                warmup_retries += 1
                                print(f'{config.secs_string(all_start)}: warmup: !! model {model.name} isnt running! [{running}]')
                                if warmup_retries < 4:
                                    print('retrying warmup...')
                                else:
                                    print('retries exhausted, skipping...')
                                    break
                            else:
                                break

                        warmup_secs = timeit.default_timer() - warmup_start
                        csv_data.append([llm_config.provider(), llm_config.model_name, '', '', '', '(warm-up)', '', '',
                                         f'{warmup_secs:.1f}', f'"{ollama_ps(model)}"', ''])
                        print(f'{config.secs_string(all_start)}: warmup done: {warmup_secs:.1f}s')
                        break
                except (Exception,) as e:
                    warmup_retries += 1
                    print(f'{config.secs_string(all_start)}: warmup attempt {warmup_retries}: Exception! '
                          f'{model.provider}:{model.name}: {e.__class__.__name__}: {e}')
                    traceback.print_exc(file=sys.stderr)
                    if warmup_retries < 4:
                        print(f"will retry in {warmup_retry_wait_secs}s")
                        time.sleep(warmup_retry_wait_secs)
                        warmup_retry_wait_secs = warmup_retries * warmup_retries
                    else:
                        break

            # settings loop
            response_line: str = ''
            for settings in CPData.llm_settings_sets[settings_set_name]:
                # todo: factory this shit
                if model.api.upper() == 'OPENAI':
                    settings.seed = run_set.seed
                    llm_config = LLMOpenAIConfig(model.name, model.provider, LLMOpenAISettings.from_settings(settings))
                elif model.api.upper() == 'ANTHROPIC':
                    settings.seed = run_set.seed
                    llm_config = LLMAnthropicConfig(model.name, model.provider, LLMAnthropicSettings.from_settings(settings))
                elif model.api.upper() == 'OLLAMA':
                    settings.ctx = run_set.ollama_ctx_size
                    settings.seed = run_set.seed
                    llm_config = LLMOllamaConfig(model.name, model.provider, LLMOllamaSettings.from_settings(settings))
                else:
                    raise ValueError(f'api must be "openai" or "anthropic" or "ollama"!')

                asyncio.run(llm_config.change_sysmsg(sysmsg_name))
                ls = timeit.default_timer()
                ls_input_tokens = 0
                ls_output_tokens = 0
                # prompts loop
                for idx, prompt_set in enumerate(CPData.llm_prompt_sets[prompt_set_name]):
                    exchange: LLMExchange | None = None
                    run_retries = 0
                    run_retry_wait_secs = 1.0
                    try:
                        while True:
                            if run_set.run_type == CPRunType.LLM:
                                exchange = CPFunctions.run_llm_prompt(prompt_set, None, llm_config, all_start)
                            elif run_set.run_type == CPRunType.RAG:
                                # todo: configure
                                exchange = CPFunctions.run_rag_prompt(prompt_set, run_set.collection_name, llm_config,
                                                                      0, 0.5, all_start)
                            if exchange is None:
                                raise ValueError(f'exchange is None!')
                            break
                    except (Exception,) as e:
                        run_retries += 1
                        print(f'{config.secs_string(all_start)} run Exception! {llm_config.provider()}:{llm_config.model_name} '
                              f'{prompt_set_name}.{prompt_set}: {e.__class__.__name__}: {e}')
                        traceback.print_exc(file=sys.stderr)
                        if run_retries < 4:
                            print(f"will retry in {run_retry_wait_secs}s")
                            time.sleep(run_retry_wait_secs)
                            warmup_retry_wait_secs = run_retries * run_retries
                        else:
                            break

                    ls_input_tokens += exchange.input_tokens
                    ls_output_tokens += exchange.output_tokens
                    # play nice with CSV: get rid of newlines and double any double-quotes
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

                if model.provider == 'OLLAMA':
                    ul_response = OllamaUtils.unload_all()
                    print(f'{config.secs_string(all_start)}: warmup: unloaded models {ul_response}')

    run_end_time = timeit.default_timer()
    print(f'{config.secs_string(all_start)}: finished run {run_set_name}/{settings_set_name}/{prompt_set_name}: '
          f'{run_end_time - run_start_time:.1f}s')


def main():
    csv_data = []

    # run(run_set_name='kf', settings_set_name='quick', sysmsg_name='professional800', prompt_set_name='galaxies4', csv_data=csv_data)
    # run(run_set_name='base', settings_set_name='quick', sysmsg_name='professional800', prompt_set_name='space', csv_data=csv_data)
    run(run_set_name='gorbash-test', settings_set_name='gorbash-test', sysmsg_name='professional800', prompt_set_name='gorbash-test', csv_data=csv_data)

    print(f'{config.secs_string(all_start)}: finished all runs: {timeit.default_timer() - all_start:.1f}s')

    # now make CSV lines from results
    print('\n\n')
    for line in csv_data:
        print(','.join(line))


if __name__ == "__main__":
    # pydevd.settrace('localhost', port=5678, stdoutToServer=True, stderrToServer=True)
    main()
