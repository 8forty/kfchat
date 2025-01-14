import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from openai.types.chat import ChatCompletion

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


@dataclass
class LLMExchange:
    prompt: str
    completion: ChatCompletion


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

    @abstractmethod
    def run_chat_completion(self, model_name: str, temp: float, top_p: float, max_tokens: int, n: int,
                            convo: Iterable[LLMExchange | tuple[str, str] | dict],
                            sysmsg: str | None = None, prompt: str | None = None) -> LLMExchange:
        pass
