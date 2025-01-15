import logging
from abc import ABC
from dataclasses import dataclass

from openai.types.chat import ChatCompletion

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMAPI(ABC):
    def __init__(self, api_type_name: str, parms: dict[str, str]):
        """

        :param api_type_name: currently: ['azure', 'ollama', 'openai', 'groq']
        :param parms: (possibly env vars) that set needed parms for the api, e.g. key, endpoint, token...
        """
        self._api_type_name = api_type_name
        self.parms = parms

    def __repr__(self) -> str:
        return f'[{self.__class__!s}:{self.__dict__!r}]'

    def type(self) -> str:
        return self._api_type_name
