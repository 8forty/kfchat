import logging

import logstuff
from llmapi import LLMAPI
from llmopenai import LLMOpenai

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMConfig:
    def __init__(self, name: str, llmapi: LLMOpenai, model_name: str, init_n: int, init_temp: float, init_top_p: float, init_max_tokens: int, init_system_message: str):
        """

        :param name
        :param llmapi:
        :param model_name:
        :param init_n:
        :param init_temp:
        :param init_top_p:
        :param init_max_tokens:
        :param init_system_message:

        """
        self.name = name
        self.llmapi = llmapi

        self.model_name = model_name
        self.n = init_n
        self.temp = init_temp
        self.top_p = init_top_p
        self.max_tokens = init_max_tokens
        self.system_message = init_system_message

    async def change_n(self, new_n: int):
        log.info(f'{self.name} changing n to: {new_n}')
        self.n = new_n

    async def change_temp(self, new_temp: float):
        log.info(f'{self.name} changing temp to: {new_temp}')
        self.temp = new_temp

    async def change_top_p(self, new_top_p: float):
        log.info(f'{self.name} changing top_p to: {new_top_p}')
        self.temp = new_top_p

    async def change_max_tokens(self, new_max_tokens: int):
        log.info(f'{self.name} changing max_tokens to: {new_max_tokens}')
        self.max_tokens = new_max_tokens

    async def change_sysmsg(self, new_system_message: str):
        log.info(f'{self.name} changing system message to: {new_system_message}')
        self.system_message = new_system_message
