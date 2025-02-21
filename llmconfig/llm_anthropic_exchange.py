import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import logstuff
from llmconfig.llmexchange import LLMExchange, LLMResponse

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMAnthropicExchange(LLMExchange):
    def __init__(self, prompt: str):
        super().__init__(prompt)

    @abstractmethod
    def responses(self) -> list[LLMResponse]:
        return []
