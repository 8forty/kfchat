import logging
from typing import Generator

from nicegui import run

import config
import logstuff
from chatexchanges import ChatExchanges, ChatExchange
from llmconfig.llmconfig import LLMConfig
from vectorstore.vsapi import VSAPI

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class InstanceData:
    _next_id: int = 1

    def __init__(self, all_llm_configs: dict[str, LLMConfig], init_llm_name: str, vectorstore: VSAPI, parms: dict[str, str]):
        self._id = InstanceData._next_id
        InstanceData._next_id += 1

        # llm stuff
        self._llm_mode: str = 'llm'
        self._llm_mode_prefix: str = 'llm: '
        self.all_llm_configs = all_llm_configs
        self.llm_config = all_llm_configs[init_llm_name]
        self.source_llm_title: str = self.llm_source(self.llm_config)  # title is "prefix: provider.model-name"

        # vs stuff
        self.vs_mode: str = 'vs'
        self.vs_mode_prefix: str = 'vs: '
        self.vectorstore = vectorstore

        # mode & source info
        self.mode: str = self._llm_mode
        self.source: str = self.source_llm_title  # always start with llm

        # specials
        self.unknown_special_prefix: str = 'unknown special command'
        self.unknown_special_message: str | None = None

        self._parms: dict[str, str] = parms
        self._chat_exchanges: ChatExchanges = ChatExchanges(config.chat_exchanges_circular_list_count)

        # from special commands
        self._info_messages: list[str] = []
        self._last_prompt: str | None = None

    def mode_is_llm(self) -> bool:
        return self.mode == self._llm_mode

    def mode_is_vs(self) -> bool:
        return self.mode == self.vs_mode

    def llm_source(self, llm_config: LLMConfig) -> str:
        """
        build full llm source title from an LLMConfig
        :param llm_config:
        :return: title: "<prefix>: <provider>.<model_name>"
        """
        return f'{self._llm_mode_prefix}{llm_config.provider()}.{llm_config.model_name}'

    def vs_source(self, collection_name: str) -> str:
        """
        build full vs source title from a vs collection name
        :param collection_name:
        :return: title: "<prefix>: <collection-name>"
        """
        return f'{self.vs_mode_prefix}{collection_name}'

    def chat_exchanges(self) -> Generator[ChatExchange, None, None]:
        for exchange in self._chat_exchanges.list():
            yield exchange

    def chat_exchange_id(self):
        return self._chat_exchanges.id()

    def clear_exchanges(self):
        self._chat_exchanges.clear()

    def add_chat_exchange(self, chat_exchange: ChatExchange) -> None:
        self._chat_exchanges.append(chat_exchange)

    def max_chat_exchanges(self) -> int:
        return self._chat_exchanges.max_exchanges()

    def info_messages(self) -> Generator[str, None, None]:
        for info_message in self._info_messages:
            yield info_message

    def clear_info_messages(self):
        self._info_messages.clear()

    def add_info_message(self, message: str) -> None:
        self._info_messages.append(message)

    def count_info_messages(self):
        return len(self._info_messages)

    def last_prompt(self) -> str:
        return self._last_prompt

    def last_prompt_update(self):
        if self._chat_exchanges.len() > 0:
            self._last_prompt = self._chat_exchanges.list()[-1].prompt

    async def change_source(self, selected_title: str):
        log.info(f'Changing source to: {selected_title}')

        if selected_title.startswith(self._llm_mode_prefix):
            self.mode = self._llm_mode
            self.source = selected_title
            self.llm_config = self.all_llm_configs[selected_title.removeprefix(self._llm_mode_prefix)]
            log.debug(f'new llm title: {self.source} (provider: {self.llm_config.provider()})')
        else:
            self.source = selected_title
            self.mode = self.vs_mode
            await run.io_bound(lambda: self.vectorstore.switch_collection(new_collection_name=self.source.removeprefix(self.vs_mode_prefix)))
            log.debug(f'new vectorstore title: {self.source}')

        self.source = selected_title

    def all_sources(self) -> list[str]:
        sources: list[str] = [self.llm_source(llm_config) for llm_config in self.all_llm_configs.values()]
        sources.extend([f'{self.vs_source(cn)}' for cn in self.vectorstore.list_collection_names()])

        # sort alpha but with the vs sources after the llm sources
        sources.sort(key=lambda k: 'zzz' + k if k.startswith(self.vs_mode_prefix) else k)
        return sources
