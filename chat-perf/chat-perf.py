import asyncio
import logging
import os
import sys
import time
import timeit
import traceback
from dataclasses import dataclass

import ollama
import requests

import llmdata
import util

# do this before config to suppress debug messages
logging.disable(logging.INFO)

import llmconfig.llm_openai_config
from cpdata import CPData, CPTargetType, CPTarget
from cpfunctions import CPFunctions
from llmconfig.llm_anthropic_config import LLMAnthropicConfig, LLMAnthropicSettings
from llmconfig.llm_ollama_config import LLMOllamaConfig, LLMOllamaSettings
from llmconfig.llm_openai_config import LLMOpenAIConfig, LLMOpenAISettings
from llmconfig.llmexchange import LLMExchange
from ollamautils import OllamaUtils

all_start = timeit.default_timer()


@dataclass
class Run:
    """
    :ivar targets:
    :ivar settings_list_name:
    :ivar sysmsg_name:
    :ivar prompts:
    """
    targets: str  # model, collection, run-type(LLM/VS/RAG)
    settings_list_name: str  # llm/vs settings
    sysmsg_name: str  # system message
    prompts: str  # prompts

    @classmethod
    def ollama_model_info(cls, model_spec: llmdata.ModelSpec, target: CPTarget) -> str:
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
                           f'{target.ctx_size},'
                           f'{OllamaUtils.get_context_length(model_name)},'
                           f'{float(model_run_size) / (1024.0 * 1024.0 * 1024.0):.1f},'
                           f'{float(vram) / (1024.0 * 1024.0 * 1024.0):.1f},'
                           f'{load} ')

        return retval

    @classmethod
    def llamacpp_model_info(cls, model_spec: llmdata.ModelSpec, target: CPTarget) -> str:
        # {'object': 'list',
        #   'data': [{'id': 'c:/llama.cpp/gemma-3-1b-it-Q4_K_M.gguf', 'object': 'model', 'created': 1747247114,
        #     'owned_by': 'llamacpp', 'meta': {'vocab_type': 1, 'n_vocab': 262144, 'n_ctx_train': 32768, 'n_embd': 1152,
        #     'n_params': 999885952, 'size': 799525120}}]}
        endpoint = (llmconfig.llm_openai_config.providers_config['LLAMACPP']['kfLLAMACPP_LLAMASERVER_ENDPOINT'].format(f'{model_spec.name}')
                    + '/models')
        response = requests.get(endpoint)
        retval = ''
        for info in response.json()['data']:
            model_id = info['id']  # e.g. z:/huggingface.co/converted/gemma-3-4b-it-Q4_K_M.gguf
            # model_spec.name e.g. gemma-3-4b-it-Q4_K_M.gguf-fa
            if model_spec.name.lower() in model_id.lower() or str(os.path.basename(model_id)).lower() in model_spec.name.lower():
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
                           f'{target.ctx_size},'
                           f'{model_ctx},'
                           f'{run_size},'
                           f'{vram},'
                           f'{load} ')

        if retval == '':
            print(f'!!!! llamacpp_model_info: no model info found! {model_spec.name}: {response.json()["data"]}')
        return retval

    @classmethod
    def ollama_warmup(cls, model: llmdata.ModelSpec, max_retries: int = 8) -> bool:
        model_name = model.name

        # the only way to clear GPU memory of any previous ollama runs
        # https://github.com/ollama/ollama/issues/10597#issuecomment-2887586741
        OllamaUtils.kill_ollama_servers()

        warmup_retries = 0
        warmup_retry_wait_secs = 1.0
        warmup_done = False
        print(f'{util.secs_string(all_start)}: warmup {model.provider}.{model_name}...')
        llm_config = LLMOllamaConfig(model_name, model.provider,
                                     LLMOllamaSettings.from_settings(CPData.llm_settings_lists['llamacpp-warmup'][0]))

        while True:
            try:
                # attempt to load the model
                llm_config.load(model_name, max_rate_limit_retries=max_retries)

                # check that the correct model is running
                if llm_config.is_model_running(model_name):
                    if warmup_retries > 0:
                        print(f'{util.secs_string(all_start)}: warmup succeeded on retry {warmup_retries}')
                    warmup_done = True
                    break
                else:
                    running = [m.name for m in ollama.ps().models]
                    warmup_retries += 1
                    print(f'{util.secs_string(all_start)}: warmup: !! model {model_name} isnt running! [{running}]')
                    if warmup_retries < max_retries:
                        print(f'{util.secs_string(all_start)}: retrying warmup...')
                        time.sleep(warmup_retry_wait_secs)
                        warmup_retry_wait_secs = warmup_retries * warmup_retries
                        continue
                    else:
                        print(f'{util.secs_string(all_start)}: retries exhausted, returning')
                        break

            except ConnectionError as e:
                warmup_retries += 1
                print(f'{util.secs_string(all_start)}: warmup attempt {warmup_retries}: Connection Exception! '
                      f'{model.provider}:{model_name}: {e.__class__.__name__}: {e}')
                # traceback.print_exc(file=sys.stderr)
                if warmup_retries < max_retries:
                    print(f'{util.secs_string(all_start)}: will retry in {warmup_retry_wait_secs}s')
                    time.sleep(warmup_retry_wait_secs)
                    warmup_retry_wait_secs = warmup_retries * warmup_retries
                continue
            except (Exception,) as e:
                warmup_retries += 1
                print(f'{util.secs_string(all_start)}: warmup attempt {warmup_retries}: Exception! '
                      f'{model.provider}:{model_name}: {e.__class__.__name__}: {e}')
                traceback.print_exc(file=sys.stderr)
                raise e

        return warmup_done

    @classmethod
    def llamacpp_warmup(cls, model: llmdata.ModelSpec, max_retries: int = 8) -> bool:
        model_name = model.name

        warmup_retries = 0
        warmup_retry_wait_secs = 1.0
        warmup_done = False
        print(f'{util.secs_string(all_start)}: warmup {model.provider}.{model_name}...')
        llm_config = LLMOpenAIConfig(model_name, model.provider,
                                     LLMOpenAISettings.from_settings(CPData.llm_settings_lists['ollama-warmup'][0]))

        while True:
            try:
                CPFunctions.run_llm_prompt(CPData.llm_prompt_lists['wakeup'][0], None, llm_config, all_start)
                warmup_done = True
                break

            except ConnectionError as e:
                warmup_retries += 1
                print(f'{util.secs_string(all_start)}: warmup attempt {warmup_retries}: Connection Exception! '
                      f'{model.provider}:{model_name}: {e.__class__.__name__}: {e}')
                # traceback.print_exc(file=sys.stderr)
                if warmup_retries < max_retries:
                    print(f'{util.secs_string(all_start)}: will retry in {warmup_retry_wait_secs}s')
                    time.sleep(warmup_retry_wait_secs)
                    warmup_retry_wait_secs = warmup_retries * warmup_retries
                continue
            except (Exception,) as e:
                warmup_retries += 1
                print(f'{util.secs_string(all_start)}: warmup attempt {warmup_retries}: Exception! '
                      f'{model.provider}:{model_name}: {e.__class__.__name__}: {e}')
                traceback.print_exc(file=sys.stderr)
                raise e

        return warmup_done

    @classmethod
    def run(cls, target_list_name: str, settings_list_name: str, sysmsg_name: str, prompt_list_name: str,
            csv_data: list[list[str]]) -> ():
        """

        :param target_list_name: type, model, collection...
        :param settings_list_name: llm/vs settings
        :param sysmsg_name: system message for llm
        :param prompt_list_name: prompt(s) to run
        :param csv_data: where to put hte csv output
        """
        run_start_time = timeit.default_timer()
        target: CPTarget
        csv_data.append(['provider', 'model', 'temp', 'max_tokens', 'sysmsg', 'prompt-set', 'tokens-in', 'tokens-out',
                         'warmup-secs', 'run-secs', 'parmsB', 'quant', 'modelGB', 'run-ctxt', 'model-ctxt',
                         'run-sizeGB', 'vramGB', 'cpu/gpu', 'last-response-1line'])

        # run-setup: targets loop
        total_warmup_secs = -1.0
        total_run_secs = -1.0
        for rs_idx, target in enumerate(CPData.target_lists[target_list_name]):
            print(f'{util.secs_string(all_start)}: running target {target_list_name}[{rs_idx}] '
                  f'{target.type} {target.model.provider} {target.model.name}...')

            warmup_secs = -1.0
            if target.type in [CPTargetType.LLM, CPTargetType.RAG]:
                model = target.model
                model_name = model.name
                settings: CPData.CPLLMSettings

                # warmup the model if necessary
                warmup_start = timeit.default_timer()
                if model.provider == 'OLLAMA':
                    if not cls.ollama_warmup(model):
                        print(f'{util.secs_string(all_start)}: retries exhausted, skipping...')
                        continue
                    warmup_secs = timeit.default_timer() - warmup_start
                elif model.provider == 'LLAMACPP':
                    if not cls.llamacpp_warmup(model):
                        print(f'{util.secs_string(all_start)}: retries exhausted, skipping...')
                        continue
                    warmup_secs = timeit.default_timer() - warmup_start

                if warmup_secs > -1:
                    total_warmup_secs += warmup_secs
                    print(f'{util.secs_string(all_start)}: warmup done: {warmup_secs:.0f}s')

                # settings loop
                response_line: str = ''
                for settings_idx, settings in enumerate(CPData.llm_settings_lists[settings_list_name]):
                    print(f'{util.secs_string(all_start)}: running settings {settings_list_name}[{settings_idx}] ')
                    # todo: factory this shit
                    if model.api.upper() == 'OPENAI':
                        settings.seed = target.seed
                        llm_config = LLMOpenAIConfig(model_name, model.provider, LLMOpenAISettings.from_settings(settings))
                    elif model.api.upper() == 'ANTHROPIC':
                        settings.seed = target.seed
                        llm_config = LLMAnthropicConfig(model_name, model.provider, LLMAnthropicSettings.from_settings(settings))
                    elif model.api.upper() == 'OLLAMA':
                        settings.ctx = target.ctx_size
                        settings.seed = target.seed
                        llm_config = LLMOllamaConfig(model_name, model.provider, LLMOllamaSettings.from_settings(settings))
                    else:
                        raise ValueError(f'api must be "openai" or "anthropic" or "ollama"!')

                    # update values in the default config
                    if sysmsg_name is not None:
                        asyncio.run(llm_config.change_sysmsg(sysmsg_name))
                    # todo: enum?
                    if model.api.upper() == 'OLLAMA':
                        # adjust the context length in the target if it's too long for the model
                        model_ctx_length = OllamaUtils.get_context_length(model_name)
                        if model_ctx_length < target.ctx_size:
                            print(f'{util.secs_string(all_start)}: adjusting context-length to model {model_ctx_length}')
                            target.ctx_size = model_ctx_length
                        # olc: LLMOllamaConfig = llm_config
                        asyncio.run(llm_config.change_ctx(target.ctx_size))

                    # prompts loop
                    ploop_start = timeit.default_timer()
                    ploop_input_tokens = 0
                    ploop_output_tokens = 0
                    for prompts_idx, prompt_list in enumerate(CPData.llm_prompt_lists[prompt_list_name]):
                        print(f'{util.secs_string(all_start)}: running prompts {prompt_list_name}[{prompts_idx}] ')
                        prompts_start = timeit.default_timer()
                        exchange: LLMExchange | None = None
                        run_retries = 0
                        run_retry_wait_secs = 1.0
                        try:
                            while True:
                                try:
                                    if target.type == CPTargetType.LLM:
                                        exchange = CPFunctions.run_llm_prompt(prompt_list, None, llm_config, all_start)
                                    elif target.type == CPTargetType.RAG:
                                        # todo: configure
                                        exchange = CPFunctions.run_rag_prompt(prompt_list, target.collection_name, llm_config,
                                                                              0, 0.5, all_start)
                                    if exchange is None:
                                        raise ValueError(f'exchange is None!')
                                    break
                                except (Exception,) as e:
                                    run_retries += 1
                                    print(f'{util.secs_string(all_start)}: run prompt Exception! {llm_config.provider()}:{llm_config.model_name} '
                                          f'{prompt_list_name}.{prompt_list}: {e.__class__.__name__}: {e}')
                                    traceback.print_exc(file=sys.stderr)
                                    if run_retries < 8:
                                        print(f'{util.secs_string(all_start)}: will retry in {run_retry_wait_secs}s')
                                        time.sleep(run_retry_wait_secs)
                                        run_retry_wait_secs = run_retries * run_retries
                                    else:
                                        raise e
                        except (Exception,):
                            break

                        if exchange.input_tokens is not None:
                            ploop_input_tokens += exchange.input_tokens
                        else:
                            print(f'{util.secs_string(all_start)}: !!! {model.provider}.{model_name}->{prompt_list_name}[{prompts_idx}] input_tokens is None!')
                        if exchange.output_tokens is not None:
                            ploop_output_tokens += exchange.output_tokens
                        else:
                            print(f'{util.secs_string(all_start)}: !!! {model.provider}.{model_name}->{prompt_list_name}[{prompts_idx}] output_tokens is None!')

                        # play nice with CSV: get rid of newlines and double any double-quotes
                        response_line = (str(exchange.responses[0].content).replace("\n", "  ")
                                         .replace('"', '""'))
                        print(f'{util.secs_string(all_start)}: {model.provider}.{model_name}:{prompt_list_name}[{prompts_idx}]: '
                              f'{exchange.input_tokens}->{exchange.output_tokens} '
                              f'{timeit.default_timer() - prompts_start:.1f}s  {response_line}')

                    ms_end = timeit.default_timer()
                    print(f'{util.secs_string(all_start)}: {model.provider}.{model_name}:{prompt_list_name}: '
                          f'[{llm_config.provider()}:{llm_config.model_name}] '
                          f'{llm_config.settings().value('temp')}/{llm_config.settings().value('max_tokens')} '
                          f'{llm_config.settings().value('system_message_name')}: '
                          f'{ploop_input_tokens}+{ploop_output_tokens} '
                          f'{timeit.default_timer() - ploop_start:.1f}s')
                    # if model.provider == 'OLLAMA':
                    #     print(f'{config.secs_string(all_start)}: ollama ps: {ollama_model_info(model, target)}')

                    # csv
                    model_info = ''
                    if llm_config.provider() == 'OLLAMA':
                        model_info = cls.ollama_model_info(model, target)
                    elif llm_config.provider() == 'LLAMACPP':
                        model_info = cls.llamacpp_model_info(model, target)

                    run_secs = ms_end - ploop_start
                    total_run_secs += run_secs
                    csv_data.append([llm_config.provider(), llm_config.model_name,
                                     str(llm_config.settings().value('temp')),
                                     str(llm_config.settings().value('max_tokens')),
                                     str(llm_config.settings().value('system_message_name')),
                                     f'{prompt_list_name}',
                                     str(ploop_input_tokens), str(ploop_output_tokens),
                                     f'{warmup_secs:.1f}',
                                     f'{run_secs:.1f}',
                                     f'{model_info}',
                                     f'"{response_line}"']
                                    )

                    if model.provider == 'OLLAMA':
                        llm_config.unload(model_name)

        total_secs = timeit.default_timer() - run_start_time
        print(f'{util.secs_string(all_start)}: finished run {target_list_name}/{settings_list_name}/{prompt_list_name}: '
              f'{total_secs:.1f}s')

        return total_warmup_secs, total_run_secs, total_secs


def main():
    # runs = [
    #     Run('kf', 'quick', 'professional800', 'space'),
    #     Run('base', 'quick', 'professional800', 'space'),
    #     Run('kf', '.7:800:2048:empty', 'empty', 'benchmark-awesome-prompts-20'),
    # ]
    #
    # runs = [Run('ollama-llama3.3:70b', '.7:800:2048:empty', 'empty', 'space'), ]
    # runs = [Run('llamacpp-gemma3-1b', '.7:800:2048:empty', 'empty', 'space'), ]
    #
    # runs = [Run('ollama-base11', '.7:800:2048:empty', 'empty', 'benchmark-awesome-prompts-20'), ]
    # runs = [
    #     Run('llamacpp-base11', '.7:800:2048:empty', 'empty', 'benchmark-awesome-prompts-20'),
    #     Run('llamacpp-fa-base11', '.7:800:2048:empty', 'empty', 'benchmark-awesome-prompts-20'),
    # ]
    runs = [Run('llamacpp-base5g', '.7:800:2048:empty', 'empty', 'space'), ]

    csv_data = []
    total_warmup_secs = 0.0
    total_run_secs = 0.0
    for run in runs:
        run_warmup_secs, run_run_secs, run_total_secs = (
            run.run(target_list_name=run.targets,  # model, collection, run-type
                    settings_list_name=run.settings_list_name,  # llm/vs settings
                    sysmsg_name=run.sysmsg_name,
                    prompt_list_name=run.prompts,
                    csv_data=csv_data))
        total_warmup_secs += run_warmup_secs
        total_run_secs += run_run_secs

        print('\n')
        for line in csv_data:
            print(','.join(line))
        print('\n')

    print(f'{util.secs_string(all_start)}: finished all runs: {timeit.default_timer() - all_start:.1f}s  '
          f'(warmups: {total_warmup_secs:.1f}s, runs: {total_run_secs:.1f}s)')

    # final CSV
    print('\n\n')
    for line in csv_data:
        print(','.join(line))


if __name__ == "__main__":
    # pydevd.settrace('localhost', port=5678, stdoutToServer=True, stderrToServer=True)
    main()
