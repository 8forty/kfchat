import logging

from nicegui import run

import config
import logstuff
from chatexchanges import ChatExchanges
from llmconfig.llmconfig import LLMConfig
from vectorstore.vsapi import VSAPI

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class InstanceData:
    _next_id: int = 1

    def __init__(self, all_llm_configs: dict[str, LLMConfig], llm_config: LLMConfig, vectorstore: VSAPI, parms: dict[str, str]):
        self._id = InstanceData._next_id
        InstanceData._next_id += 1
        self.parms: dict[str, str] = parms
        self.exchanges: ChatExchanges = ChatExchanges(config.chat_exchanges_circular_list_count)

        # from special commands
        self.info_messages: list[str] = []
        self.last_prompt: str | None = None
        self.unknown_special_prefix: str = 'unknown special command'
        self.unknown_special_message: str | None = None

        # llm stuff
        self.llm_mode: str = 'llm'
        self.llm_mode_prefix: str = 'llm: '
        self.all_llm_configs = all_llm_configs
        self.llm_config = llm_config
        self.source_llm_title: str = self.llm_source(self.llm_config)  # title is "prefix: provider.model-name"

        # vs stuff
        self.vs_mode: str = 'vs'
        self.vs_mode_prefix: str = 'vs: '
        self.vectorstore = vectorstore

        # mode & source info
        self.mode: str = self.llm_mode
        self.source: str = self.source_llm_title  # always start with llm

    def mode_is_llm(self) -> bool:
        return self.mode == self.llm_mode

    def mode_is_vs(self) -> bool:
        return self.mode == self.vs_mode

    def llm_source(self, llm_config: LLMConfig) -> str:
        """
        build full llm source title from an LLMConfig
        :param llm_config:
        :return: title: "<prefix>: <provider>.<model_name>"
        """
        return f'{self.llm_mode_prefix}{llm_config.provider()}.{llm_config.model_name}'

    def vs_source(self, collection_name: str) -> str:
        """
        build full vs source title from a vs collection name
        :param collection_name:
        :return: title: "<prefix>: <collection-name>"
        """
        return f'{self.vs_mode_prefix}{collection_name}'

    def clear(self):
        self.exchanges.clear()

    async def change_source(self, selected_title: str):
        log.info(f'Changing source to: {selected_title}')

        if selected_title.startswith(self.llm_mode_prefix):
            self.mode = self.llm_mode
            self.source = selected_title
            self.llm_config = self.all_llm_configs[selected_title.removeprefix(self.llm_mode_prefix)]
            log.debug(f'new llm title: {self.source} (provider: {self.llm_config.provider()})')
        else:
            self.source = selected_title
            self.mode = self.vs_mode
            await run.io_bound(lambda: self.vectorstore.switch_collection(new_collection_name=self.source.removeprefix(self.vs_mode_prefix)))
            log.debug(f'new vectorstore title: {self.source}')

        self.source = selected_title

    async def change_n(self, new_n: int):
        for llm_config in self.all_llm_configs.values():
            await llm_config.change_n(new_n)

    async def change_temp(self, new_temp: float):
        for llm_config in self.all_llm_configs.values():
            await llm_config.change_temp(new_temp)

    async def change_top_p(self, new_top_p: float):
        for llm_config in self.all_llm_configs.values():
            await llm_config.change_top_p(new_top_p)

    async def change_max_tokens(self, new_max_tokens: int):
        for llm_config in self.all_llm_configs.values():
            await llm_config.change_max_tokens(new_max_tokens)

    async def change_sysmsg(self, new_system_message_name: str):
        for llm_config in self.all_llm_configs.values():
            await llm_config.change_sysmsg(new_system_message_name)

    def all_sources(self) -> list[str]:
        sources: list[str] = [self.llm_source(llm_config) for llm_config in self.all_llm_configs.values()]
        sources.extend([f'{self.vs_source(cn)}' for cn in self.vectorstore.list_collection_names()])

        # sort alpha but with the vs sources after the llm sources
        sources.sort(key=lambda k: 'zzz' + k if k.startswith(self.vs_mode_prefix) else k)
        return sources
