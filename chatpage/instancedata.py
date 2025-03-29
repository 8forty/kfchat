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
        self._all_llm_configs = all_llm_configs
        self._llm_config = all_llm_configs[init_llm_name]
        self._source_llm_title: str = self.llm_source(self._llm_config)  # title is "prefix: provider.model-name"

        # vs stuff
        self._vs_mode: str = 'vs'
        self._vs_mode_prefix: str = 'vs: '
        self._vectorstore = vectorstore

        # mode & source info
        self._mode: str = self._llm_mode
        self._source: str = self._source_llm_title  # always start with llm

        # special commands
        self.special_about = 'special commands: *, *info, *repeat, *clear, (n) *1/*2... '
        self._unknown_special_prefix: str = 'unknown special command'

        self._parms: dict[str, str] = parms
        self._chat_exchanges: ChatExchanges = ChatExchanges(config.chat_exchanges_circular_list_count)

        # from special commands
        self._info_messages: list[str] = []
        self._last_prompt: str | None = None

    def mode_is_llm(self) -> bool:
        return self._mode == self._llm_mode

    def mode_is_vs(self) -> bool:
        return self._mode == self._vs_mode

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
        return f'{self._vs_mode_prefix}{collection_name}'

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

    def gllm_config(self) -> LLMConfig:
        return self._llm_config

    def gmode(self) -> str:
        return self._mode

    def gsource(self) -> str:
        return self._source

    async def change_source(self, selected_title: str):
        log.info(f'Changing source to: {selected_title}')

        if selected_title.startswith(self._llm_mode_prefix):
            self._mode = self._llm_mode
            self._source = selected_title
            self._llm_config = self._all_llm_configs[selected_title.removeprefix(self._llm_mode_prefix)]
            log.debug(f'new llm title: {self._source} (provider: {self._llm_config.provider()})')
        else:
            self._source = selected_title
            self._mode = self._vs_mode
            await run.io_bound(lambda: self._vectorstore.switch_collection(new_collection_name=self._source.removeprefix(self._vs_mode_prefix)))
            log.debug(f'new vectorstore title: {self._source}')

        self._source = selected_title

    def gvectorstore(self) -> VSAPI:
        return self._vectorstore

    def all_sources(self) -> list[str]:
        sources: list[str] = [self.llm_source(llm_config) for llm_config in self._all_llm_configs.values()]
        sources.extend([f'{self.vs_source(cn)}' for cn in self._vectorstore.list_collection_names()])

        # sort alpha but with the vs sources after the llm sources
        sources.sort(key=lambda k: 'zzz' + k if k.startswith(self._vs_mode_prefix) else k)
        return sources

    def add_unknown_special_message(self, prompt: str) -> None:
        self.add_info_message(f'{self._unknown_special_prefix}: {prompt}; {self.special_about}')
