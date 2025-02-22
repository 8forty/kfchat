import logging
from abc import ABC

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMResponse(ABC):
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class LLMExchange(ABC):
    def __init__(self, prompt: str, responses: list[LLMResponse], response_duration_seconds: float, problems: dict[int, str]):
        self.prompt = prompt
        self.responses = responses
        self.response_duration_secs = response_duration_seconds
        self.problems = problems  # [response-idx, problem-description]
