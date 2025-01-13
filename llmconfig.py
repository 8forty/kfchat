import logging

import logstuff
from llmapi import LLMAPI

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMConfig:
    def __init__(self, llmapi: LLMAPI, model_name: str, default_n: int, default_temp: float, default_max_tokens: int, default_system_message: str):
        """

        :param llmapi:
        :param model_name:
        :param default_n:
        :param default_temp:
        :param default_max_tokens:
        :param default_system_message:

        """
        self.llmapi: LLMAPI = llmapi

        self.model_name: str = model_name
        self.n: int = 1
        self.temp: float = default_temp
        self.max_tokens: int = default_max_tokens
        self.system_message: str = default_system_message

    def change_n(self, new_n: int):
        log.info(f'changing n to: {new_n}')
        self.n = new_n

    def change_temp(self, new_temp: float):
        log.info(f'changing temp to: {new_temp}')
        self.temp = new_temp

    def change_max_tokens(self, new_max_tokens: int):
        log.info(f'changing max_tokens to: {new_max_tokens}')
        self.max_tokens = new_max_tokens

    def change_sysmsg(self, new_system_message: str):
        log.info(f'changing system message to: {new_system_message}')
        self.system_message = new_system_message
