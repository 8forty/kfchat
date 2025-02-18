import logging

from nicegui import run

import config
import logstuff
from chatexchanges import ChatExchanges
from llmoaiconfig import LLMOaiConfig
from vectorstore.vsapi import VSAPI

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class InstanceData:
    _next_id: int = 1

    def __init__(self, llm_configs: dict[str, LLMOaiConfig], llm_config: LLMOaiConfig, vectorstore: VSAPI, parms: dict[str, str]):
        self._id = InstanceData._next_id
        InstanceData._next_id += 1
        self.parms: dict[str, str] = parms
        self.exchanges: ChatExchanges = ChatExchanges(config.chat_exchanges_circular_list_count)

        # from special commands
        self.info_messages: list[str] = []
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

        # source info
        self.source_select_name: str = self.source_llm_name  # start with llm
        self.current_source_type: str = self.llm_source_type
        self.source_name: str = self.source_select_name  # name of the source object (we want to start with the llm, so select-name and name are the same)

    def source_type_is_llm(self) -> bool:
        return self.current_source_type == self.llm_source_type

    def source_type_is_vs(self) -> bool:
        return self.current_source_type == self.vs_source_type

    def source_api_name_llm(self, llm_config: LLMOaiConfig) -> str:
        return f'{self.llm_name_prefix}{llm_config.api_type()}:{llm_config.model_name}'

    def forget(self):
        self.exchanges.clear()

    async def change_source(self, selected_name: str):
        log.info(f'Changing source to: {selected_name}')

        if selected_name.startswith(self.llm_name_prefix):
            self.current_source_type = self.llm_source_type
            long_name = selected_name.removeprefix(self.llm_name_prefix)  # removes "llm: "
            self.source_name = ':'.join(long_name.split(':')[1:])  # removes e.g. "groq:"
            self.llm_config = self.llm_configs[self.source_name]
            log.debug(f'new llm name: {self.source_name} (api_type: {self.llm_config.api_type()})')
        else:
            self.source_name = selected_name.removeprefix(self.vs_name_prefix)
            self.current_source_type = self.vs_source_type
            await run.io_bound(lambda: self.vectorstore.switch_index(self.source_name))
            log.debug(f'new vs name: {self.source_name}')

        self.source_select_name = selected_name

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
