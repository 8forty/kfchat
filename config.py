import datetime
import logging
import random
import time
import timeit

import openai

import logstuff
from modelapi import ModelAPI

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

random.seed(27)

name = 'kfchat'

chat_exchanges_circular_list_count = 10


def now_datetime() -> str:
    return datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d-%H:%M:%S')


def now_time() -> str:
    return datetime.datetime.now(datetime.UTC).strftime('%H:%M:%S')


def secs_string(start: float, end: float = None) -> str:
    if end is None:
        end = timeit.default_timer()
    return time.strftime('%H:%M:%S', time.gmtime(end - start))


class LLMConfig:
    def __init__(self, api_type: str, env_values: dict[str, str], model_name: str, default_temp: float, max_tokens: int, system_message: str):
        """

        :param api_type: same list as ModelAPI.__init__
        :param env_values: parms for the api, e.g. key, endpoint, token...
        :param model_name:
        :param default_temp:
        :param max_tokens:
        :param system_message:

        """
        # todo: these should come from e.g. pref screen
        self.model_api: ModelAPI = ModelAPI(api_type, parms=env_values)
        self.model_name: str = model_name
        self.default_temp: float = default_temp
        self.max_tokens: int = max_tokens
        self.system_message: str = system_message
        self.client: openai.OpenAI = self.model_api.client()
