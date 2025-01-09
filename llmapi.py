import logging
from dataclasses import dataclass
from typing import Iterable

import openai
from openai.types.chat import ChatCompletion

import logstuff
from config import redact

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


@dataclass
class LLMExchange:
    prompt: str
    completion: ChatCompletion


class LLMAPI:
    def __init__(self, api_type: str, parms: dict[str, str]):
        """

        :param api_type: currently: ['azure', 'ollama', 'openai', 'groq']
        :param parms: (possibly env vars) that set needed parms for the api, e.g. key, endpoint, token...
        """
        if api_type in ['azure', 'ollama', 'openai', 'groq']:
            self.api_type = api_type
            self.parms = parms
            self.api_client = None
        else:
            raise ValueError(f'{__class__.__name__}: invalid api_type! {api_type}')

    def __repr__(self) -> str:
        return f'[{self.__class__!s}:{self.__dict__!r}]'

    def type(self) -> str:
        return self.api_type

    def client(self) -> openai.OpenAI:
        if self.api_client is not None:
            return self.api_client

        if self.api_type == "azure":
            # token_provider = azure.identity.get_bearer_token_provider(
            #     azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            # )
            # client = openai.AzureOpenAI(
            #     api_version=self.parms.get("AZURE_OPENAI_API_VERSION"),
            #     azure_endpoint=self.parms.get("AZURE_OPENAI_ENDPOINT"),
            #     azure_ad_token_provider=token_provider,
            # )
            log.info(f'building LLM API for [{self.api_type}]: {self.parms.get("AZURE_OPENAI_ENDPOINT")=}, {self.parms.get("AZURE_OPENAI_API_KEY")}, '
                     f'{redact(self.parms.get("AZURE_OPENAI_API_VERSION"))}')
            self.api_client = openai.AzureOpenAI(azure_endpoint=self.parms.get("AZURE_OPENAI_ENDPOINT"),
                                                 api_key=self.parms.get("AZURE_OPENAI_API_KEY"),
                                                 api_version=self.parms.get("AZURE_OPENAI_API_VERSION"))
        elif self.api_type == "ollama":
            log.info(f'building LLM API for [{self.api_type}]: {self.parms.get("OLLAMA_ENDPOINT")=}')
            self.api_client = openai.OpenAI(base_url=self.parms.get("OLLAMA_ENDPOINT"),
                                            api_key="nokeyneeded")
        elif self.api_type == "openai":
            log.info(f'building LLM API for [{self.api_type}]: {self.parms.get("OPENAI_ENDPOINT")}, {redact(self.parms.get("OPENAI_API_KEY"))}')
            self.api_client = openai.OpenAI(base_url=self.parms.get("OPENAI_ENDPOINT"),
                                            api_key=self.parms.get("OPENAI_API_KEY"))
        elif self.api_type == "groq":
            log.info(f'building LLM API for [{self.api_type}]: {self.parms.get("GROQ_ENDPOINT")}, {redact(self.parms.get("GROQ_API_KEY"))}')
            self.api_client = openai.OpenAI(base_url=self.parms.get("GROQ_OPENAI_ENDPOINT"),
                                            api_key=self.parms.get("GROQ_API_KEY"))
        elif self.api_type == "github":
            base_url = "https://models.inference.ai.azure.com"
            log.info(f'building LLM API for [{self.api_type}]: {base_url=}, {redact(self.parms.get("GITHUB_TOKEN"))}')
            self.api_client = openai.OpenAI(base_url=base_url,
                                            api_key=self.parms.get("GITHUB_TOKEN"))
        else:
            raise ValueError(f'invalid api_type! {self.api_type}')

        return self.api_client

    def run_chat_completion(self, model_name: str, temp: float, max_tokens: int, n: int,
                            convo: Iterable[LLMExchange | tuple[str, str] | dict],
                            sysmsg: str | None = None, prompt: str | None = None) -> ChatCompletion:
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
        chat_completion: ChatCompletion = self.client().chat.completions.create(
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

        return chat_completion
