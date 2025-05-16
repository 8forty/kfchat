import asyncio
import logging
import sys
import time
import timeit
import traceback
from dataclasses import dataclass

import ollama
import requests

import config
from cpdata import CPData, CPRunType, CPRunSpec
from cpfunctions import CPFunctions
from llmconfig.llm_anthropic_config import LLMAnthropicConfig, LLMAnthropicSettings
from llmconfig.llm_ollama_config import LLMOllamaConfig, LLMOllamaSettings
from llmconfig.llm_openai_config import LLMOpenAIConfig, LLMOpenAISettings
import llmconfig.llm_openai_config
from llmconfig.llmexchange import LLMExchange
from ollamautils import OllamaUtils

logging.disable(logging.INFO)

all_start = timeit.default_timer()


def ollama_model_info(model_spec: config.ModelSpec, run_spec: CPRunSpec) -> str:
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
        if model_name == model_spec.name:
            parm_size = float(ps_dict[model_name].details.parameter_size[:-1])
            if ps_dict[model_name].details.parameter_size[-1] == 'M':
                parmsb: int = int(round(parm_size / 1000.0, 1))
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
                       f'{run_spec.ctx_size},'
                       f'{OllamaUtils.get_context_length(model_name)},'
                       f'{float(model_run_size) / (1024.0 * 1024.0 * 1024.0):.1f},'
                       f'{float(vram) / (1024.0 * 1024.0 * 1024.0):.1f},'
                       f'{load} ')

    return retval


def llamacpp_model_info(model_spec: config.ModelSpec, run_spec: CPRunSpec) -> str:
    # {'object': 'list',
    #   'data': [{'id': 'c:/llama.cpp/gemma-3-1b-it-Q4_K_M.gguf', 'object': 'model', 'created': 1747247114,
    #     'owned_by': 'llamacpp', 'meta': {'vocab_type': 1, 'n_vocab': 262144, 'n_ctx_train': 32768, 'n_embd': 1152,
    #     'n_params': 999885952, 'size': 799525120}}]}
    endpoint = (llmconfig.llm_openai_config.providers_config['LLAMACPP']['kfLLAMACPP_ENDPOINT'].format(f'{model_spec.name}')
                + '/models')
    response = requests.get(endpoint)
    retval = ''
    for info in response.json()['data']:
        model_id = info['id']
        if model_spec.name in model_id:
            parmsb: int = int(round(int(info['meta']['n_params']) / 1000000000.0, 1))
            quant: str = ''
            model_base_size: float = float(info['meta']['size']) / (1024.0 * 1024.0 * 1024.0)
            model_ctx: int = int(info['meta']['n_ctx_train'])
            run_size = ''
            vram = ''
            load: str = ''
            retval += (f'{parmsb},'
                       f'{quant},'
                       f'{model_base_size:.1f},'
                       f'{run_spec.ctx_size},'
                       f'{model_ctx},'
                       f'{run_size},'
                       f'{vram},'
                       f'{load} ')

    return retval


def ollama_warmup(model: config.ModelSpec, max_retries: int = 8) -> bool:
    model_name = model.name

    # the only way to GPU memory of any previous ollama runs
    OllamaUtils.kill_ollama_servers()

    warmup_retries = 0
    warmup_retry_wait_secs = 1.0
    warmup_done = False
    print(f'{config.secs_string(all_start)}: warmup {model.provider}.{model_name}...')
    llm_config = LLMOllamaConfig(model_name, model.provider,
                                 LLMOllamaSettings.from_settings(CPData.llm_settings_sets['llamacpp-warmup'][0]))

    while True:
        try:
            # attempt to load the model
            llm_config.load(model_name, max_rate_limit_retries=max_retries)

            # check that the correct model is running
            if llm_config.is_model_running(model_name):
                if warmup_retries > 0:
                    print(f'{config.secs_string(all_start)}: warmup succeeded on retry {warmup_retries}')
                warmup_done = True
                break
            else:
                running = [m.name for m in ollama.ps().models]
                warmup_retries += 1
                print(f'{config.secs_string(all_start)}: warmup: !! model {model_name} isnt running! [{running}]')
                if warmup_retries < max_retries:
                    print(f'{config.secs_string(all_start)}: retrying warmup...')
                    time.sleep(warmup_retry_wait_secs)
                    warmup_retry_wait_secs = warmup_retries * warmup_retries
                    continue
                else:
                    print(f'{config.secs_string(all_start)}: retries exhausted, returning')
                    break

        except ConnectionError as e:
            warmup_retries += 1
            print(f'{config.secs_string(all_start)}: warmup attempt {warmup_retries}: Connection Exception! '
                  f'{model.provider}:{model_name}: {e.__class__.__name__}: {e}')
            # traceback.print_exc(file=sys.stderr)
            if warmup_retries < max_retries:
                print(f'{config.secs_string(all_start)}: will retry in {warmup_retry_wait_secs}s')
                time.sleep(warmup_retry_wait_secs)
                warmup_retry_wait_secs = warmup_retries * warmup_retries
            continue
        except (Exception,) as e:
            warmup_retries += 1
            print(f'{config.secs_string(all_start)}: warmup attempt {warmup_retries}: Exception! '
                  f'{model.provider}:{model_name}: {e.__class__.__name__}: {e}')
            traceback.print_exc(file=sys.stderr)
            raise e

    return warmup_done


def llamacpp_warmup(model: config.ModelSpec, max_retries: int = 8) -> bool:
    model_name = model.name

    warmup_retries = 0
    warmup_retry_wait_secs = 1.0
    warmup_done = False
    print(f'{config.secs_string(all_start)}: warmup {model.provider}.{model_name}...')
    llm_config = LLMOpenAIConfig(model_name, model.provider,
                                 LLMOpenAISettings.from_settings(CPData.llm_settings_sets['ollama-warmup'][0]))

    while True:
        try:
            CPFunctions.run_llm_prompt(CPData.llm_prompt_sets['wakeup'][0], None, llm_config, all_start)
            warmup_done = True
            break

        except ConnectionError as e:
            warmup_retries += 1
            print(f'{config.secs_string(all_start)}: warmup attempt {warmup_retries}: Connection Exception! '
                  f'{model.provider}:{model_name}: {e.__class__.__name__}: {e}')
            # traceback.print_exc(file=sys.stderr)
            if warmup_retries < max_retries:
                print(f'{config.secs_string(all_start)}: will retry in {warmup_retry_wait_secs}s')
                time.sleep(warmup_retry_wait_secs)
                warmup_retry_wait_secs = warmup_retries * warmup_retries
            continue
        except (Exception,) as e:
            warmup_retries += 1
            print(f'{config.secs_string(all_start)}: warmup attempt {warmup_retries}: Exception! '
                  f'{model.provider}:{model_name}: {e.__class__.__name__}: {e}')
            traceback.print_exc(file=sys.stderr)
            raise e

    return warmup_done


def run(run_specs_name: str, settings_set_name: str, sysmsg_name: str, prompt_set_name: str, csv_data: list[list[str]]):
    """

    :param run_specs_name: model, collection, run-type
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

    # run-setup: run specs loop
    for rs_idx, run_spec in enumerate(CPData.run_specs[run_specs_name]):
        print(f'{config.secs_string(all_start)}: running run-spec {run_specs_name}[{rs_idx}] '
              f'{run_spec.run_type} {run_spec.model.provider} {run_spec.model.name}...')

        if run_spec.run_type in [CPRunType.LLM, CPRunType.RAG]:
            model = run_spec.model
            model_name = model.name
            settings: CPData.LLMRawSettings

            # warmup the model if necessary
            warmup_secs = -1
            warmup_start = timeit.default_timer()
            if model.provider == 'OLLAMA':
                if not ollama_warmup(model):
                    print(f'{config.secs_string(all_start)}: retries exhausted, skipping...')
                    continue
                warmup_secs = timeit.default_timer() - warmup_start
            elif model.provider == 'LLAMACPP':
                if not llamacpp_warmup(model):
                    print(f'{config.secs_string(all_start)}: retries exhausted, skipping...')
                    continue
                warmup_secs = timeit.default_timer() - warmup_start

            if warmup_secs > -1:
                print(f'{config.secs_string(all_start)}: warmup done: {warmup_secs:.0f}s')

            # settings loop
            response_line: str = ''
            for settings_idx, settings in enumerate(CPData.llm_settings_sets[settings_set_name]):
                print(f'{config.secs_string(all_start)}: running settings {settings_set_name}[{settings_idx}] ')
                # todo: factory this shit
                if model.api.upper() == 'OPENAI':
                    settings.seed = run_spec.seed
                    llm_config = LLMOpenAIConfig(model_name, model.provider, LLMOpenAISettings.from_settings(settings))
                elif model.api.upper() == 'ANTHROPIC':
                    settings.seed = run_spec.seed
                    llm_config = LLMAnthropicConfig(model_name, model.provider, LLMAnthropicSettings.from_settings(settings))
                elif model.api.upper() == 'OLLAMA':
                    settings.ctx = run_spec.ctx_size
                    settings.seed = run_spec.seed
                    llm_config = LLMOllamaConfig(model_name, model.provider, LLMOllamaSettings.from_settings(settings))
                else:
                    raise ValueError(f'api must be "openai" or "anthropic" or "ollama"!')

                # update values in the default config
                if sysmsg_name is not None:
                    asyncio.run(llm_config.change_sysmsg(sysmsg_name))
                # todo: enum?
                if model.api.upper() == 'OLLAMA':
                    # adjust the context length in the run_spec if it's too long for the model
                    model_ctx_length = OllamaUtils.get_context_length(model_name)
                    if model_ctx_length < run_spec.ctx_size:
                        print(f'{config.secs_string(all_start)}: adjusting context-length to model {model_ctx_length}')
                        run_spec.ctx_size = model_ctx_length
                    # olc: LLMOllamaConfig = llm_config
                    asyncio.run(llm_config.change_ctx(run_spec.ctx_size))

                # prompts loop
                ploop_start = timeit.default_timer()
                ploop_input_tokens = 0
                ploop_output_tokens = 0
                for prompts_idx, prompt_set in enumerate(CPData.llm_prompt_sets[prompt_set_name]):
                    print(f'{config.secs_string(all_start)}: running prompts {prompt_set_name}[{prompts_idx}] ')
                    prompts_start = timeit.default_timer()
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
                        if run_retries < 8:
                            print(f'{config.secs_string(all_start)}: will retry in {run_retry_wait_secs}s')
                            time.sleep(run_retry_wait_secs)
                            run_retry_wait_secs = run_retries * run_retries
                        else:
                            break

                    ploop_input_tokens += exchange.input_tokens
                    ploop_output_tokens += exchange.output_tokens
                    # play nice with CSV: get rid of newlines and double any double-quotes
                    response_line = (str(exchange.responses[0].content).replace("\n", "  ")
                                     .replace('"', '""'))
                    print(f'{config.secs_string(all_start)}: {model.provider}.{model_name}:{prompt_set_name}[{prompts_idx}]: '
                          f'{exchange.input_tokens}->{exchange.output_tokens} '
                          f'{timeit.default_timer() - prompts_start:.1f}s  {response_line}')

                ms_end = timeit.default_timer()
                print(f'{config.secs_string(all_start)}: {model.provider}.{model_name}:{prompt_set_name}: '
                      f'[{llm_config.provider()}:{llm_config.model_name}] '
                      f'{llm_config.settings().value('temp')}/{llm_config.settings().value('max_tokens')} '
                      f'{llm_config.settings().value('system_message_name')}: '
                      f'{ploop_input_tokens}+{ploop_output_tokens} '
                      f'{timeit.default_timer() - ploop_start:.1f}s')
                if model.provider == 'OLLAMA':
                    print(f'{config.secs_string(all_start)}: ollama ps: {ollama_model_info(model, run_spec)}')

                # csv
                model_info = ''
                if llm_config.provider() == 'OLLAMA':
                    model_info = ollama_model_info(model, run_spec)
                elif llm_config.provider() == 'LLAMACPP':
                    model_info = llamacpp_model_info(model, run_spec)

                csv_data.append([llm_config.provider(), llm_config.model_name,
                                 str(llm_config.settings().value('temp')),
                                 str(llm_config.settings().value('max_tokens')),
                                 str(llm_config.settings().value('system_message_name')),
                                 f'{prompt_set_name}',
                                 str(ploop_input_tokens), str(ploop_output_tokens),
                                 f'{warmup_secs:.1f}',
                                 f'{ms_end - ploop_start:.1f}',
                                 f'{model_info}',
                                 f'"{response_line}"']
                                )

                if model.provider == 'OLLAMA':
                    llm_config.unload(model_name)

    run_end_time = timeit.default_timer()
    print(f'{config.secs_string(all_start)}: finished run {run_specs_name}/{settings_set_name}/{prompt_set_name}: '
          f'{run_end_time - run_start_time:.1f}s')


@dataclass
class RunSet:
    """
    :ivar cprun_specs_name:
    :ivar settings_set_name:
    :ivar sysmsg_name:
    :ivar prompt_set_name:
    """
    cprun_specs_name: str  # model, collection, run-type(LLM/VS/RAG)
    settings_set_name: str  # llm/vs settings
    sysmsg_name: str  # system message
    prompt_set_name: str  # prompts


def main():
    run_sets = {
        'quick': RunSet('kf', 'quick', 'professional800', 'space'),
        'base': RunSet('base', 'quick', 'professional800', 'space'),

        'kf': RunSet('kf', '.7:800:2048:empty', 'empty', 'benchmark-awesome-prompts-20'),

        'ollama-bm20-gemma1b': RunSet('ollama-gemma3-1b', '.7:800:2048:empty', 'empty', 'benchmark-awesome-prompts-20'),
        'ollama-space-gemma1b': RunSet('ollama-gemma3-1b', '.7:800:2048:empty', 'empty', 'space'),
        'ollama-space-ll70': RunSet('ollama-llama3.3:70b', '.7:800:2048:empty', 'empty', 'space'),

        'llamacpp-bm20-gemma1b': RunSet('llamacpp-gemma3-1b', '.7:800:2048:empty', 'empty', 'benchmark-awesome-prompts-20'),
        'llamacpp-space-gemma1b': RunSet('llamacpp-gemma3-1b', '.7:800:2048:empty', 'empty', 'space'),
        'llamacpp-space-gemma4b': RunSet('llamacpp-gemma3-4b', '.7:800:2048:empty', 'empty', 'space'),
        'llamacpp-space-gemma12b': RunSet('llamacpp-gemma3-12b', '.7:800:2048:empty', 'empty', 'space'),
        'llamacpp-space-gemma27b': RunSet('llamacpp-gemma3-27b', '.7:800:2048:empty', 'empty', 'space'),

        'bm20-gemma': RunSet('ollama-gemma', '.7:800:2048:empty', 'empty', 'benchmark-awesome-prompts-20'),
        'bm20-llama3.2': RunSet('ollama-llama3.2', '.7:800:2048:empty', 'empty', 'benchmark-awesome-prompts-20'),
        'bm20-llama3.3': RunSet('ollama-llama3.3', '.7:800:2048:empty', 'empty', 'benchmark-awesome-prompts-20'),
        'bm20-phi': RunSet('ollama-phi', '.7:800:2048:empty', 'empty', 'benchmark-awesome-prompts-20'),
        'bm20-qwen': RunSet('ollama-qwen', '.7:800:2048:empty', 'empty', 'benchmark-awesome-prompts-20'),
        'bm20-other': RunSet('ollama-other', '.7:800:2048:empty', 'empty', 'benchmark-awesome-prompts-20'),

        'ollama-bm20-base11': RunSet('ollama-base11', '.7:800:2048:empty', 'empty', 'benchmark-awesome-prompts-20'),

        'llamacpp-space-base11': RunSet('llamacpp-base11', '.7:800:2048:empty', 'empty', 'space'),
        'llamacpp-bm20-base11': RunSet('llamacpp-base11', '.7:800:2048:empty', 'empty', 'benchmark-awesome-prompts-20'),
    }

    # run_set_names = ['quick', 'base', 'kf',]

    # run_set_names = ['ollama-space-ll70',]
    # run_set_names = ['llamacpp-space-gemma1b', ]
    # run_set_names = ['llamacpp-space-gemma27b', ]

    # run_set_names = ['ollama-bm20-base11', ]
    run_set_names = ['llamacpp-bm20-base11', ]

    csv_data = []
    for rsn in run_set_names:
        run(run_specs_name=run_sets[rsn].cprun_specs_name,  # model, collection, run-type
            settings_set_name=run_sets[rsn].settings_set_name,  # llm/vs settings
            sysmsg_name=run_sets[rsn].sysmsg_name,
            prompt_set_name=run_sets[rsn].prompt_set_name,
            csv_data=csv_data)

        print('\n')
        for line in csv_data:
            print(','.join(line))
        print('\n')

    print(f'{config.secs_string(all_start)}: finished all run-sets: {timeit.default_timer() - all_start:.1f}s')

    # final CSV
    print('\n\n')
    for line in csv_data:
        print(','.join(line))


if __name__ == "__main__":
    # pydevd.settrace('localhost', port=5678, stdoutToServer=True, stderrToServer=True)
    main()
