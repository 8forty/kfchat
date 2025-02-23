import logging

from anthropic.types import Message

import logstuff
from llmconfig.llmexchange import LLMExchange, LLMMessagePair
from llmconfig.llmsettings import LLMSettings

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMAnthropicExchange(LLMExchange):
    def __init__(self, prompt: str, message: Message,
                 provider: str,
                 model_name: str,
                 settings: LLMSettings,
                 response_duration_seconds: float):
        stop_problem_reasons = ['max_tokens', 'tool_use']
        # todo: responses might contain type != 'text'?
        super().__init__(prompt=prompt,
                         provider=provider,
                         model_name=model_name,
                         settings=settings,
                         responses=[LLMMessagePair(message.role, tblock.text) for tblock in message.content if tblock.type == 'text'],
                         input_tokens=message.usage.prompt_tokens,
                         output_tokens=message.usage.completion_tokens,
                         response_duration_seconds=response_duration_seconds,
                         problems={-1: message.stop_reason} if message.stop_reason in stop_problem_reasons else {})
