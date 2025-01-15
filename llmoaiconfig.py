import logging
from dataclasses import dataclass
from typing import Iterable

import openai
from openai.types.chat import ChatCompletion

import logstuff
from config import redact
from llmconfig import LLMConfig

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


@dataclass
class LLMOaiExchange:
    prompt: str
    completion: ChatCompletion


class LLMOaiConfig(LLMConfig):
    def __init__(self, name: str, api_type_name: str, parms: dict[str, str], model_name: str,
                 init_n: int, init_temp: float, init_top_p: float, init_max_tokens: int, init_system_message: str):
        """

        :param name
        :param api_type_name
        :param parms
        :param model_name:
        :param init_n:
        :param init_temp:
        :param init_top_p:
        :param init_max_tokens:
        :param init_system_message:

        """
        super().__init__(name, api_type_name, parms)

        self.model_name = model_name
        self.n = init_n
        self.temp = init_temp
        self.top_p = init_top_p
        self.max_tokens = init_max_tokens
        self.system_message = init_system_message

        if self.api_type_name not in ['azure', 'ollama', 'openai', 'groq']:
            raise ValueError(f'{__class__.__name__}: invalid api_type! {api_type_name}')
        self._api_client = None

    def __repr__(self) -> str:
        return f'[{self.__class__!s}:{self.__dict__!r}]'

    async def change_n(self, new_n: int):
        log.info(f'{self.name} changing n to: {new_n}')
        self.n = new_n

    async def change_temp(self, new_temp: float):
        log.info(f'{self.name} changing temp to: {new_temp}')
        self.temp = new_temp

    async def change_top_p(self, new_top_p: float):
        log.info(f'{self.name} changing top_p to: {new_top_p}')
        self.temp = new_top_p

    async def change_max_tokens(self, new_max_tokens: int):
        log.info(f'{self.name} changing max_tokens to: {new_max_tokens}')
        self.max_tokens = new_max_tokens

    async def change_sysmsg(self, new_system_message: str):
        log.info(f'{self.name} changing system message to: {new_system_message}')
        self.system_message = new_system_message

    def _client(self) -> openai.OpenAI:
        if self._api_client is not None:
            return self._api_client

        if self.api_type_name == "azure":
            # token_provider = azure.identity.get_bearer_token_provider(
            #     azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            # )
            # client = openai.AzureOpenAI(
            #     api_version=self.parms.get("AZURE_OPENAI_API_VERSION"),
            #     azure_endpoint=self.parms.get("AZURE_OPENAI_ENDPOINT"),
            #     azure_ad_token_provider=token_provider,
            # )
            log.info(f'building LLM API for [{self.api_type_name}]: {self.parms.get("AZURE_OPENAI_ENDPOINT")=}, '
                     f'AZURE_OPENAI_API_KEY={redact(self.parms.get("AZURE_OPENAI_API_KEY"))}, '
                     f'{self.parms.get("AZURE_OPENAI_API_VERSION")=}')
            self._api_client = openai.AzureOpenAI(azure_endpoint=self.parms.get("AZURE_OPENAI_ENDPOINT"),
                                                  api_key=self.parms.get("AZURE_OPENAI_API_KEY"),
                                                  api_version=self.parms.get("AZURE_OPENAI_API_VERSION"))
        elif self.api_type_name == "ollama":
            log.info(f'building LLM API for [{self.api_type_name}]: {self.parms.get("OLLAMA_ENDPOINT")=}')
            self._api_client = openai.OpenAI(base_url=self.parms.get("OLLAMA_ENDPOINT"),
                                             api_key="nokeyneeded")
        elif self.api_type_name == "openai":
            log.info(f'building LLM API for [{self.api_type_name}]: {self.parms.get("OPENAI_ENDPOINT")=}, '
                     f'OPENAI_API_KEY={redact(self.parms.get("OPENAI_API_KEY"))}')
            self._api_client = openai.OpenAI(base_url=self.parms.get("OPENAI_ENDPOINT"),
                                             api_key=self.parms.get("OPENAI_API_KEY"))
        elif self.api_type_name == "groq":
            log.info(f'building LLM API for [{self.api_type_name}]: {self.parms.get("GROQ_ENDPOINT")=}, '
                     f'GROQ_API_KEY={redact(self.parms.get("GROQ_API_KEY"))}')
            self._api_client = openai.OpenAI(base_url=self.parms.get("GROQ_OPENAI_ENDPOINT"),
                                             api_key=self.parms.get("GROQ_API_KEY"))
        elif self.api_type_name == "github":
            base_url = "https://models.inference.ai.azure.com"
            log.info(f'building LLM API for [{self.api_type_name}]: {base_url=}, {redact(self.parms.get("GITHUB_TOKEN"))}')
            self._api_client = openai.OpenAI(base_url=base_url,
                                             api_key=self.parms.get("GITHUB_TOKEN"))
        else:
            raise ValueError(f'invalid api type! {self.api_type_name}')

        return self._api_client

    def run_chat_completion(self, model_name: str, temp: float, top_p: float, max_tokens: int, n: int,
                            convo: Iterable[LLMOaiExchange | tuple[str, str] | dict],
                            sysmsg: str | None = None, prompt: str | None = None) -> LLMOaiExchange:
        """
        run chat-completion
        :param model_name:
        :param temp:
        :param top_p:
        :param max_tokens:
        :param n:
        :param convo: properly ordered list of either LLMOpenaiExchange's or tuples of (role, value) ; tuples must include system message and prompt
        :param sysmsg: used as sysmsg iff convo is provided in LLMOpenaiExchange objects, otherwise the prompt should be in the tuples of convo
        :param prompt: used as prompt iff convo is provided in LLMOpenaiExchange objects, otherwise the prompt should be in the tuples of convo
        :return: results as an OpenAI ChatCompletion object
        """
        messages: list[dict] = []
        if sysmsg is not None:
            # add the system message
            if sysmsg is not None and len(sysmsg) > 0:
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

        # todo: seed, etc. (by actual llm?)
        log.debug(f'{model_name=}, {temp=}, {top_p=}, {max_tokens=}, {n=}, {sysmsg=} {prompt=}')
        chat_completion: ChatCompletion = self._client().chat.completions.create(
            model=model_name,
            temperature=temp,  # default 1.0, 0.0->2.0
            top_p=top_p,  # default 1, ~0.01->1.0
            messages=messages,
            max_tokens=max_tokens,  # default 16?
            n=n,

            stream=False,  # todo: allow streaming

            # seed=27,
            # frequency_penalty=1,  # default 0, -2.0->2.0
            # presence_penalty=1,  # default 0, -2.0->2.0
            # stop=[],
        )
        return LLMOaiExchange(prompt, chat_completion)
