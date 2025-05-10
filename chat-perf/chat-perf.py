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


def ollama_ps(model_spec: config.ModelSpec, run_spec: CPRunSpec) -> str:
    if model_spec.provider == 'OLLAMA':
        # ProcessResponse(models=[
        # Model(model='llama3.2:1b', name='llama3.2:1b',
        #   digest='baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878',
        #   expires_at=datetime.datetime(2025, 5, 6, 20, 39, 13, 620726, tzinfo=TzInfo(-07:00)),
        #   size=8584751104, size_vram=8584751104,
        #   details=ModelDetails(parent_model='', format='gguf', family='llama', families=['llama'], parameter_size='1.2B',
        #     quantization_level='Q8_0')),
        # ...
        # ])
        ps_dict = {m.model: m for m in ollama.ps().models}

        retval = ''
        for model_name in ps_dict.keys():

            parm_size = float(ps_dict[model_name].details.parameter_size[:-1])
            if ps_dict[model_name].details.parameter_size[-1] == 'M':
                parmsb: int = int(round(parm_size / 1024.0, 1))
            else:
                parmsb = int(parm_size)
            model_run_size = ps_dict[model_name].size
            vram = ps_dict[model_name].size_vram
            gpu = float(vram) / float(model_run_size) if model_run_size > vram else 1.0
            cpu = 1.0 - gpu
            load = f'{cpu * 100.0:.0f}%/{gpu * 100.0:.0f}% CPU/GPU' if cpu > 0.0 else f'100% GPU*'
            # load += f' (vsize: {float(ps_dict[model_name].size) / (1024.0 * 1024.0 * 1024.0):.1f} vram: {float(vram) / (1024.0 * 1024.0 * 1024.0):.1f} model: {float(model_size) / (1024.0 * 1024.0 * 1024.0):.1f})'
            retval += (f'{parmsb},'
                       f'{ps_dict[model_name].details.quantization_level},'
                       f'{float(OllamaUtils.get_model_base_size(model_name)) / (1024.0 * 1024.0 * 1024.0):.1f},'
                       f'{run_spec.ollama_ctx_size},{OllamaUtils.get_context_length(model_name)},'
                       f'{float(model_run_size) / (1024.0 * 1024.0 * 1024.0):.1f},{float(vram) / (1024.0 * 1024.0 * 1024.0):.1f},'
                       f'{load} ')

        return retval
    else:
        return ''


def run(run_set_name: str, settings_set_name: str, sysmsg_name: str, prompt_set_name: str, csv_data: list[list[str]]):
    """

    :param run_set_name: model, collection, run-type
    :param settings_set_name: llm/vs settings
    :param sysmsg_name: system message for llm
    :param prompt_set_name: prompt(s) to run
    :param csv_data: where to put hte csv output
    """
    run_start_time = timeit.default_timer()
    run_spec: CPRunSpec
    csv_data.append(['provider', 'model', 'temp', 'max_tokens', 'sysmsg', 'prompt-set', 'tokens-in', 'tokens-out',
                     'warmup-secs', 'run-secs', 'parmsB', 'quant', 'modelGB', 'run-ctxt', 'model-ctxt',
                     'run-sizeGB', 'vramGB', 'cpu/gpu', 'last-response-1line'])

    # loop run specs from run_set
    for run_spec in CPData.run_sets[run_set_name]:
        print(f'{config.secs_string(all_start)}: running {run_spec.run_type} {run_spec.model.provider} {run_spec.model.name}...')

        if run_spec.run_type in [CPRunType.LLM, CPRunType.RAG]:
            model = run_spec.model
            model_name = model.name
            settings: CPData.LLMRawSettings

            # warmup the model if necessary
            warmup_secs = 0
            if model.provider == 'OLLAMA':
                warmup_retries = 0
                warmup_retry_wait_secs = 1.0
                warmup_done = False
                warmup_start = timeit.default_timer()
                try:
                    print(f'{config.secs_string(all_start)}: warmup {model.provider} {model_name}...')
                    llm_config = LLMOllamaConfig(model_name, model.provider,
                                                 LLMOllamaSettings.from_settings(CPData.llm_settings_sets['ollama-warmup'][0]))

                    # until ollama reports the model as running
                    while True:
                        llm_config.load(model_name)

                        # check that the correct model is running
                        if not llm_config.is_model_running(model_name):
                            running = [m.name for m in ollama.ps().models]
                            warmup_retries += 1
                            print(f'{config.secs_string(all_start)}: warmup: !! model {model_name} isnt running! [{running}]')
                            if warmup_retries < 4:
                                print(f'{config.secs_string(all_start)}: retrying warmup...')
                                continue
                            else:
                                print(f'{config.secs_string(all_start)}: retries exhausted, skipping...')
                                break
                        else:
                            warmup_done = True
                            break

                    if warmup_done:
                        warmup_secs = timeit.default_timer() - warmup_start
                        print(f'{config.secs_string(all_start)}: warmup done: {warmup_secs:.0f}s')
                    else:
                        continue  # next run_spec
                except (Exception,) as e:
                    warmup_retries += 1
                    print(f'{config.secs_string(all_start)}: warmup attempt {warmup_retries}: Exception! '
                          f'{model.provider}:{model_name}: {e.__class__.__name__}: {e}')
                    traceback.print_exc(file=sys.stderr)
                    if warmup_retries < 4:
                        print(f'{config.secs_string(all_start)}: will retry in {warmup_retry_wait_secs}s')
                        time.sleep(warmup_retry_wait_secs)
                        warmup_retry_wait_secs = warmup_retries * warmup_retries
                    else:
                        break

            # settings loop
            response_line: str = ''
            for settings in CPData.llm_settings_sets[settings_set_name]:
                # todo: factory this shit
                if model.api.upper() == 'OPENAI':
                    settings.seed = run_spec.seed
                    llm_config = LLMOpenAIConfig(model_name, model.provider, LLMOpenAISettings.from_settings(settings))
                elif model.api.upper() == 'ANTHROPIC':
                    settings.seed = run_spec.seed
                    llm_config = LLMAnthropicConfig(model_name, model.provider, LLMAnthropicSettings.from_settings(settings))
                elif model.api.upper() == 'OLLAMA':
                    settings.ctx = run_spec.ollama_ctx_size
                    settings.seed = run_spec.seed
                    llm_config = LLMOllamaConfig(model_name, model.provider, LLMOllamaSettings.from_settings(settings))
                else:
                    raise ValueError(f'api must be "openai" or "anthropic" or "ollama"!')

                # update values in the default config
                asyncio.run(llm_config.change_sysmsg(sysmsg_name))
                # todo: enum?
                if model.api.upper() == 'OLLAMA':
                    # adjust the context length in the run_spec if it's too long for the model
                    model_ctx_length = OllamaUtils.get_context_length(model_name)
                    if model_ctx_length < run_spec.ollama_ctx_size:
                        print(f'{config.secs_string(all_start)}: adjusting context-length to model {model_ctx_length}')
                        run_spec.ollama_ctx_size = model_ctx_length
                    # olc: LLMOllamaConfig = llm_config
                    asyncio.run(llm_config.change_ctx(run_spec.ollama_ctx_size))

                # prompts loop
                ls = timeit.default_timer()
                ls_input_tokens = 0
                ls_output_tokens = 0
                for idx, prompt_set in enumerate(CPData.llm_prompt_sets[prompt_set_name]):
                    exchange: LLMExchange | None = None
                    run_retries = 0
                    run_retry_wait_secs = 1.0
                    try:
                        while True:
                            if run_spec.run_type == CPRunType.LLM:
                                exchange = CPFunctions.run_llm_prompt(prompt_set, None, llm_config, all_start)
                            elif run_spec.run_type == CPRunType.RAG:
                                # todo: configure
                                exchange = CPFunctions.run_rag_prompt(prompt_set, run_spec.collection_name, llm_config,
                                                                      0, 0.5, all_start)
                            if exchange is None:
                                raise ValueError(f'exchange is None!')
                            break
                    except (Exception,) as e:
                        run_retries += 1
                        print(f'{config.secs_string(all_start)}: run Exception! {llm_config.provider()}:{llm_config.model_name} '
                              f'{prompt_set_name}.{prompt_set}: {e.__class__.__name__}: {e}')
                        traceback.print_exc(file=sys.stderr)
                        if run_retries < 4:
                            print(f'{config.secs_string(all_start)}: will retry in {run_retry_wait_secs}s')
                            time.sleep(run_retry_wait_secs)
                            run_retry_wait_secs = run_retries * run_retries
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
                print(f'{config.secs_string(all_start)}: ollama ps: {ollama_ps(model, run_spec)}')

                # csv
                csv_data.append([llm_config.provider(), llm_config.model_name,
                                 str(llm_config.settings().value('temp')),
                                 str(llm_config.settings().value('max_tokens')),
                                 str(llm_config.settings().value('system_message_name')),
                                 f'{prompt_set_name}',
                                 str(ls_input_tokens), str(ls_output_tokens),
                                 f'{warmup_secs:.1f}',
                                 f'{ms_end - ls:.1f}',
                                 f'{ollama_ps(model, run_spec)}',
                                 f'"{response_line}"']
                                )

                if model.provider == 'OLLAMA':
                    # print(subprocess.run(['ollama', 'ps'], capture_output=True, text=True).stdout)
                    llm_config.unload(model_name)
                    OllamaUtils.kill_ollama_servers()

    run_end_time = timeit.default_timer()
    print(f'{config.secs_string(all_start)}: finished run {run_set_name}/{settings_set_name}/{prompt_set_name}: '
          f'{run_end_time - run_start_time:.1f}s')


def main():
    csv_data = []

    # run(run_set_name='kf', settings_set_name='quick', sysmsg_name='professional800', prompt_set_name='galaxies4', csv_data=csv_data)
    # run(run_set_name='base', settings_set_name='quick', sysmsg_name='professional800', prompt_set_name='space', csv_data=csv_data)
    run(run_set_name='kf',  # 'gorbash-test-fast-ones-gg1',  # model, collection, run-type
        settings_set_name='gorbash-test',  # llm/vs settings
        sysmsg_name='professional800',
        prompt_set_name='awesome-chatgpt-prompts',  # 'benchmark-prompts',  # 'gorbash-compliance-hotline',
        # prompt_set_name='gorbash-security',
        csv_data=csv_data)

    print(f'{config.secs_string(all_start)}: finished all runs: {timeit.default_timer() - all_start:.1f}s')

    # now make CSV lines from results
    print('\n\n')
    for line in csv_data:
        print(','.join(line))


if __name__ == "__main__":
    # pydevd.settrace('localhost', port=5678, stdoutToServer=True, stderrToServer=True)
    main()
