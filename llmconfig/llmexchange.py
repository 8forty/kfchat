import logging
from abc import ABC

import logstuff
from llmconfig.llmsettings import LLMSettings

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMResponse(ABC):
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def __repr__(self) -> str:
        return f'[{self.__class__!s}:{self.__dict__!r}]'


class LLMExchange(ABC):
    def __init__(self, prompt: str, provider: str, model_name: str, settings: LLMSettings,
                 responses: list[LLMResponse], input_tokens: int, output_tokens: int,
                 response_duration_seconds: float, problems: dict[int, str]):
        self.prompt = prompt
        self.provider = provider
        self.model_name = model_name
        self.settings = settings
        self.responses = responses
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.response_duration_secs = response_duration_seconds
        self.problems = problems  # [response-idx, problem-description]

    def __repr__(self) -> str:
        return f'[{self.__class__!s}:{self.__dict__!r}]'

