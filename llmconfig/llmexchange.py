import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


@dataclass
class LLMResponse:
    role: str
    content: str


class LLMExchange(ABC):
    def __init__(self, prompt: str):
        self.prompt = prompt

    @abstractmethod
    def responses(self) -> list[LLMResponse]:
        pass
