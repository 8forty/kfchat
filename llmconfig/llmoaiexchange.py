import logging

from openai.types.chat import ChatCompletion

import logstuff
from llmconfig.llmexchange import LLMExchange, LLMResponse

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMOaiExchange(LLMExchange):
    def __init__(self, prompt: str, chat_completion: ChatCompletion, response_duration_seconds: float):
        stop_problem_reasons = ['length', 'too many tokens', 'content_filter', 'flagged by content filter', 'tool_calls', 'called a tool', 'function_call']
        super().__init__(prompt=prompt,
                         responses=[LLMResponse(choice.message.role, choice.message.content) for choice in chat_completion.choices],
                         response_duration_seconds=response_duration_seconds,
                         problems={idx: choice.finish_reason for idx, choice in enumerate(chat_completion.choices) if choice.finish_reason in stop_problem_reasons})
        self.cc = chat_completion
