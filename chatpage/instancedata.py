import logging

from nicegui import run, ui
from nicegui.elements.input import Input
from nicegui.elements.scroll_area import ScrollArea
from nicegui.elements.spinner import Spinner

import config
import logstuff
from chatexchanges import ChatExchanges
from llmoaiconfig import LLMOaiConfig
from vsapi import VSAPI

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class InstanceData:
    _next_id: int = 1

    def __init__(self, llm_configs: dict[str, LLMOaiConfig], llm_config: LLMOaiConfig, vectorstore: VSAPI, env_values: dict[str, str]):
        self._id = InstanceData._next_id
        InstanceData._next_id += 1

        self.env_values: dict[str, str] = env_values
        self.info_messages: list[str] = []
        self.exchanges: ChatExchanges = ChatExchanges(config.chat_exchanges_circular_list_count)
        self.last_prompt: str | None = None

        # llm stuff
        self.llm_source_type: str = 'llm'
        self.llm_name_prefix: str = 'llm: '
        self.llm_configs = llm_configs
        self.llm_config = llm_config
        self.source_llm_name: str = self.source_api_name_llm(self.llm_config)

        # vs stuff
        self.vs_source_type: str = 'vs'
        self.vs_name_prefix: str = 'vs: '
        self.vectorstore = vectorstore

        # #### source info
        self.source_select_name: str = self.source_llm_name
        self.source_name: str = self.source_select_name  # name of the source object (we want to start with the llm, so select-name and name are the same)
        self.source_api: VSAPI | None = None  # the current VS api, or None for any llm

    def source_type(self) -> str:
        return self.llm_source_type if self.source_api is None else self.vs_source_type

    def source_api_name_llm(self, llm_config: LLMOaiConfig) -> str:
        return f'{self.llm_name_prefix}{llm_config.name}:{llm_config.model_name}'

    def forget(self):
        self.exchanges.clear()

    async def change_source(self, selected_name: str, spinner: Spinner, prompt_input: Input):
        log.info(f'Changing source to: {selected_name}')
        prompt_input.disable()
        spinner.set_visibility(True)

        try:
            if selected_name.startswith(self.llm_name_prefix):
                self.source_api = None
                self.source_name = selected_name.removeprefix(self.llm_name_prefix)
                self.llm_config = self.llm_configs[self.source_name.split(':')[0]]
            else:
                self.source_name = selected_name.removeprefix(self.vs_name_prefix)
                self.source_api = self.vectorstore
                await run.io_bound(self.vectorstore.change_index, self.source_name)

            self.source_select_name = selected_name
        except (Exception,) as e:
            errmsg = f'change source failed! {e}'
            log.warning(errmsg)
            ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)

        prompt_input.enable()
        spinner.set_visibility(False)

    async def change_n(self, new_n: int):
        for llm_config in self.llm_configs.values():
            await llm_config.change_n(new_n)

    async def change_temp(self, new_temp: float):
        for llm_config in self.llm_configs.values():
            await llm_config.change_temp(new_temp)

    async def change_top_p(self, new_top_p: float):
        for llm_config in self.llm_configs.values():
            await llm_config.change_top_p(new_top_p)

    async def change_max_tokens(self, new_max_tokens: int):
        for llm_config in self.llm_configs.values():
            await llm_config.change_max_tokens(new_max_tokens)

    async def change_sysmsg(self, new_system_message_name: str):
        for llm_config in self.llm_configs.values():
            await llm_config.change_sysmsg(new_system_message_name)

    def source_names_list(self) -> list[str]:
        source_names: list[str] = [self.source_api_name_llm(llm_config) for llm_config in self.llm_configs.values()]
        source_names.extend([f'{self.vs_name_prefix}{name}' for name in self.vectorstore.list_index_names()])
        source_names.sort(key=lambda k: 'zzz' + k if k.startswith(self.vs_name_prefix) else k)  # sort with the vs sources after the llm sources
        return source_names

    @staticmethod
    def pp(resp: str) -> str:
        return f'{resp}'

    async def refresh_instance(self, scroller: ScrollArea) -> None:
        # todo: local-storage-session to separate messages
        scroller.clear()
        with scroller:
            # info-messages are not exchanges/responses, e.g. they come from special commands
            if len(self.info_messages) > 0:
                for im in self.info_messages:
                    ui.label(im).classes('w-full text-left')
                self.info_messages.clear()
            elif self.exchanges.len() > 0:
                self.last_prompt = self.exchanges.list()[-1].prompt
                response_text_classes = 'w-full text-lg text-green text-left px-10'
                response_subscript_classes = 'w-full italic text-xs text-black text-left px-10'

                for exchange in self.exchanges.list():

                    # the prompt
                    ui.label(exchange.prompt).classes('w-full font-bold text-lg text-blue text-left px-10')

                    # the response(s)
                    with (ui.column().classes('w-full gap-y-0')):
                        results: list[str] = []
                        subscript_results_info: list[list[str]] = []  # a list of metric strings per result
                        subscript_context_info = ''
                        subscript_extra_info: list[str] = []

                        # llm response
                        if exchange.llm_response is not None:
                            ex_resp = exchange.llm_response
                            for choice in ex_resp.chat_completion.choices:
                                results.append(f'{choice.message.content}')  # .classes(response_text_classes)
                                subscript_results_info.append([f'logprobs: {choice.logprobs}'])
                            subscript_context_info += f'{self.llm_source_type},{ex_resp.api_type}:{ex_resp.model_name},n:{ex_resp.n},temp:{ex_resp.temp},top_p:{ex_resp.top_p},max_tokens:{ex_resp.max_tokens}'
                            subscript_extra_info.append(f'tokens:{ex_resp.chat_completion.usage.prompt_tokens}->{ex_resp.chat_completion.usage.completion_tokens}')
                            subscript_extra_info.append(f'{self.llm_config.system_message}')

                        # vector store response
                        if exchange.vector_store_response is not None:
                            subscript_context_info += f'{self.vs_source_type},{self.source_name}'
                            for result in exchange.vector_store_response.results:
                                results.append(f'[{self.vs_source_type}]: {result.content}')  # .classes(response_text_classes)

                                metric_list = []
                                for metric in result.metrics:
                                    val = result.metrics[metric]
                                    if val is not None and len(str(val)) > 0:
                                        if isinstance(val, float):
                                            metric_list.append(f'{metric}: {result.metrics[metric]:.03f}')
                                        else:
                                            metric_list.append(f'{metric}: {result.metrics[metric]}')
                                subscript_results_info.append(metric_list)

                        # results
                        for ri in range(0, len(results)):
                            for line in self.pp(results[ri]).split('\n'):
                                ui.label(line).classes(response_text_classes)
                            for rinfo in subscript_results_info[ri]:
                                ui.label(rinfo).classes(response_subscript_classes)

                        # entire-response extra stuff
                        ui.label(f'[{subscript_context_info}]: {exchange.response_duration_secs:.1f}s').classes(response_subscript_classes)
                        for ei in subscript_extra_info:
                            ui.label(f'{ei}').classes(response_subscript_classes)

                        # stop problems
                        stop_problems_string = ''
                        for choice_idx, stop_problem in exchange.stop_problems().items():
                            stop_problems_string += f'stop[{choice_idx}]:{stop_problem}'
                        if len(stop_problems_string) > 0:
                            ui.label(f'{stop_problems_string}').classes('w-full italic text-xs text-red text-left px-10')

                        # exchange problems
                        if exchange.overflowed():
                            ui.label(f'exchange history (max:{self.exchanges.max_exchanges()}) overflowed!  Oldest exchange dropped'
                                     ).classes('w-full italic text-xs text-red text-left px-10')
            else:
                ui.label('No messages yet').classes('mx-auto my-36 absolute-center text-2xl italic')

        scroller.scroll_to(percent=1.1)
