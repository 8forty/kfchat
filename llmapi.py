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
    def _client(self):
        pass

    def run_chat_completion(self, model_name: str, temp: float, max_tokens: int, n: int,
                            convo: Iterable[LLMExchange | tuple[str, str] | dict],
                            sysmsg: str | None = None, prompt: str | None = None) -> LLMExchange:
        """
        run chat-completion
        :param model_name:
        :param temp:
        :param max_tokens:
        :param n:
        :param convo: properly ordered list of either LLMExchange's or tuples of (role, value) ; tuples must include system message and prompt
        :param sysmsg: used as sysmsg iff convo is provided in LLMExchange objects, otherwise the prompt should be in the tuples of convo
        :param prompt: used as prompt iff convo is provided in LLMExchange objects, otherwise the prompt should be in the tuples of convo
        :return: results as an OpenAI ChatCompletion object
        """
        messages: list[dict] = []
        if sysmsg is not None:
            # add the system message
            if sysmsg is not None:
                messages = [{'role': 'system', 'content': sysmsg}]

            # add the convo
            for exchange in convo:
                # todo: what about previous vector-store responses?
                messages.append({'role': 'user', 'content': exchange.prompt})
                for choice in exchange.completion.choices:
                    messages.append({'role': choice.message.role, 'content': choice.message.content})

            # add the prompt
            messages.append({'role': 'user', 'content': prompt})
        else:
            # transform convo to list-of-dicts, elements are either tuples or already dicts (and I guess a mix of each, why not?)
            messages = [{t[0]: t[1]} if isinstance(t, tuple) else t for t in convo]

        # todo: seed, top_p, etc. (by actual llm?)
        chat_completion: ChatCompletion = self._client().chat.completions.create(
            model=model_name,
            temperature=temp,  # default 1.0, 0.0->2.0
            messages=messages,
            max_tokens=max_tokens,  # default 16?
            n=n,

            stream=False,  # todo: allow streaming

            # seed=27,
            # top_p=1,  # default 1, ~0.01->1.0
            # frequency_penalty=1,  # default 0, -2.0->2.0
            # presence_penalty=1,  # default 0, -2.0->2.0
            # stop=[],
        )
        return LLMExchange(prompt, chat_completion)
