import logging
from typing import Iterable

import openai
from openai.types.chat import ChatCompletion

import logstuff
from config import redact
from llmapi import LLMAPI, LLMExchange

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMOpenai(LLMAPI):
    def __init__(self, api_type_name: str, parms: dict[str, str]):
        """
        for any client that supports the well-known OpenAI api
        :param api_type_name: currently: ['azure', 'ollama', 'openai', 'groq']
        :param parms: (possibly env vars) that set needed parms for the api, e.g. key, endpoint, token...
        """
        super().__init__(api_type_name, parms)
        if self._api_type_name not in ['azure', 'ollama', 'openai', 'groq']:
            raise ValueError(f'{__class__.__name__}: invalid api_type! {api_type_name}')
        self._api_client = None

    def __repr__(self) -> str:
        return f'[{self.__class__!s}:{self.__dict__!r}]'

    def _client(self) -> openai.OpenAI:
        if self._api_client is not None:
            return self._api_client

        if self._api_type_name == "azure":
            # token_provider = azure.identity.get_bearer_token_provider(
            #     azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            # )
            # client = openai.AzureOpenAI(
            #     api_version=self.parms.get("AZURE_OPENAI_API_VERSION"),
            #     azure_endpoint=self.parms.get("AZURE_OPENAI_ENDPOINT"),
            #     azure_ad_token_provider=token_provider,
            # )
            log.info(f'building LLM API for [{self._api_type_name}]: {self.parms.get("AZURE_OPENAI_ENDPOINT")=}, '
                     f'AZURE_OPENAI_API_KEY={redact(self.parms.get("AZURE_OPENAI_API_KEY"))}, '
                     f'{self.parms.get("AZURE_OPENAI_API_VERSION")=}')
            self._api_client = openai.AzureOpenAI(azure_endpoint=self.parms.get("AZURE_OPENAI_ENDPOINT"),
                                                  api_key=self.parms.get("AZURE_OPENAI_API_KEY"),
                                                  api_version=self.parms.get("AZURE_OPENAI_API_VERSION"))
        elif self._api_type_name == "ollama":
            log.info(f'building LLM API for [{self._api_type_name}]: {self.parms.get("OLLAMA_ENDPOINT")=}')
            self._api_client = openai.OpenAI(base_url=self.parms.get("OLLAMA_ENDPOINT"),
                                             api_key="nokeyneeded")
        elif self._api_type_name == "openai":
            log.info(f'building LLM API for [{self._api_type_name}]: {self.parms.get("OPENAI_ENDPOINT")=}, '
                     f'OPENAI_API_KEY={redact(self.parms.get("OPENAI_API_KEY"))}')
            self._api_client = openai.OpenAI(base_url=self.parms.get("OPENAI_ENDPOINT"),
                                             api_key=self.parms.get("OPENAI_API_KEY"))
        elif self._api_type_name == "groq":
            log.info(f'building LLM API for [{self._api_type_name}]: {self.parms.get("GROQ_ENDPOINT")=}, '
                     f'GROQ_API_KEY={redact(self.parms.get("GROQ_API_KEY"))}')
            self._api_client = openai.OpenAI(base_url=self.parms.get("GROQ_OPENAI_ENDPOINT"),
                                             api_key=self.parms.get("GROQ_API_KEY"))
        elif self._api_type_name == "github":
            base_url = "https://models.inference.ai.azure.com"
            log.info(f'building LLM API for [{self._api_type_name}]: {base_url=}, {redact(self.parms.get("GITHUB_TOKEN"))}')
            self._api_client = openai.OpenAI(base_url=base_url,
                                             api_key=self.parms.get("GITHUB_TOKEN"))
        else:
            raise ValueError(f'invalid api_type! {self._api_type_name}')

        return self._api_client
