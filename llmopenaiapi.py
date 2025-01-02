import logging

import openai
from openai.types.chat import ChatCompletion

import logstuff
from chatexchanges import ChatExchanges

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMOpenaiAPI:
    def __init__(self, oai_api_client: openai.OpenAI):
        self.oai_api_client = oai_api_client

    def llm_run_prompt(self, model_name: str, temp: float, max_tokens: int, n: int,
                       sysmsg: str, prompt: str, convo: ChatExchanges) -> ChatCompletion:
        messages = [{'role': 'system', 'content': sysmsg}]
        for exchange in convo.list():
            # todo: what about previous vector-store responses?
            if exchange.llm_response is not None:
                messages.append({'role': 'user', 'content': exchange.prompt})
                for choice in exchange.llm_response.choices:
                    messages.append({'role': choice.message.role, 'content': choice.message.content})
        messages.append({'role': 'user', 'content': prompt})

        # todo: seed, top_p, etc. (by actual llm?)
        llm_response: ChatCompletion = self.oai_api_client.chat.completions.create(
            model=model_name,
            temperature=temp,  # default 1.0, 0.0->2.0
            messages=messages,
            max_tokens=max_tokens,  # default 16?
            n=n,

            stream=False,

            # seed=27,
            # top_p=1,  # default 1, ~0.01->1.0
            # frequency_penalty=1,  # default 0, -2.0->2.0
            # presence_penalty=1,  # default 0, -2.0->2.0
            # stop=[],
        )

        return llm_response
