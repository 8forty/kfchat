import logging

from openai.types.chat import ChatCompletion

import logstuff
from llmconfig.llmexchange import LLMExchange, LLMResponse

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMOaiExchange(LLMExchange):
    def __init__(self, prompt: str, chat_completion: ChatCompletion):
        super().__init__(prompt)
        self.cc = chat_completion

    def responses(self) -> list[LLMResponse]:
        return [LLMResponse(choice.message.role, choice.message.content) for choice in self.cc.choices]
