import logging
from abc import ABC, abstractmethod

import config
import logstuff
from llmconfig.llmexchange import LLMExchange, LLMMessagePair
from llmconfig.llmsettings import LLMSettings

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMConfig(ABC):

    def __init__(self, model_name: str, provider_name: str):
        """

        :param model_name
        :param provider_name
        """
        self.model_name = model_name
        self._provider = provider_name

    def __repr__(self) -> str:
        return f'{self.__class__!s}:{self.__dict__!r}'
        # return f'{self.__class__!s}:{self.result_id=!r},{self.metrics=!r},self.content="{self.content[0:20]}{"..." if len(self.content) > 20 else ""}"'

    def provider(self) -> str:
        return self._provider

    async def change_n(self, new_n: int):
        log.info(f'{self.model_name} changing n to: {new_n}')
        self.settings().n = new_n

    async def change_temp(self, new_temp: float):
        log.info(f'{self.model_name} changing temp to: {new_temp}')
        self.settings().temp = new_temp

    async def change_top_p(self, new_top_p: float):
        log.info(f'{self.model_name} changing top_p to: {new_top_p}')
        self.settings().temp = new_top_p

    async def change_max_tokens(self, new_max_tokens: int):
        log.info(f'{self.model_name} changing max_tokens to: {new_max_tokens}')
        self.settings().max_tokens = new_max_tokens

    async def change_sysmsg(self, new_system_message_name: str):
        new_system_message = config.LLMData.sysmsg_all[new_system_message_name]
        log.info(f'{self.model_name} changing system message to: {new_system_message_name}:{new_system_message}')
        self.settings().system_message_name = new_system_message_name
        self.settings().system_message = new_system_message

    @abstractmethod
    def change_settings(self, new_settings: LLMSettings) -> LLMSettings:
        """
        replaces existing settings with new settings, returns the old settings object
        :param new_settings:
        :return:
        """
        pass

    @abstractmethod
    def settings(self) -> LLMSettings:
        pass

    @abstractmethod
    def copy_settings(self) -> LLMSettings:
        pass

    @staticmethod
    def _clean_messages(messages: list[LLMMessagePair]) -> list[LLMMessagePair]:
        for message in messages:
            if message.role == 'system':
                log.warning(f"'system' message removed from messages! {message}")
        return [mp for mp in messages if mp.role != 'system']

    @abstractmethod
    def _chat(self, messages: list[LLMMessagePair], max_rate_limit_retries: int = 10) -> LLMExchange:
        pass

    def chat_messages(self, messages: list[LLMMessagePair], max_rate_limit_retries: int = 10) -> LLMExchange:
        """
        run chat-completion from a list of messages that includes the prompt as a final dict: {role': 'user', 'content': '...'}
        NOTE: this configuration's system message will be used instead of any supplied in messages
        :param messages:
        :param max_rate_limit_retries:
        """
        messages = LLMConfig._clean_messages(messages)
        log.debug(f'{messages=}')
        return self._chat(messages, max_rate_limit_retries=max_rate_limit_retries)

    def chat_convo(self, convo: list[LLMExchange], prompt: str, max_rate_limit_retries: int = 10) -> LLMExchange:
        """
        run chat-completion
        NOTE: this configuration's system message will be used instead of any supplied in convo
        :param convo: properly ordered list of LLMOpenaiExchange's
        :param prompt: the prompt duh
        :param max_rate_limit_retries:
        """
        messages: list[LLMMessagePair] = []

        # add the convo
        for exchange in convo:
            # todo: what about previous vector-store responses?
            messages.append(LLMMessagePair('user', exchange.prompt))
            messages.extend(exchange.responses)

        # add the prompt
        messages.append(LLMMessagePair('user', prompt))
        return self.chat_messages(messages, max_rate_limit_retries=max_rate_limit_retries)
