import logging
import time
from dataclasses import dataclass
from typing import Iterable

import openai
from openai.types.chat import ChatCompletion

import data
import logstuff
from config import redact
from llmconfig import LLMConfig

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


@dataclass
class LLMOaiExchange:
    prompt: str
    completion: ChatCompletion


class LLMOaiSettings:
    def __init__(self, init_n: int, init_temp: float, init_top_p: float, init_max_tokens: int, init_system_message_name: str):
        """

        :param init_n:
        :param init_temp:
        :param init_top_p:
        :param init_max_tokens:
        :param init_system_message_name:

        """
        self.n = init_n
        self.temp = init_temp
        self.top_p = init_top_p
        self.max_tokens = init_max_tokens
        self.system_message_name = init_system_message_name
        self.system_message = data.sysmsg_all[init_system_message_name]


class LLMOaiConfig(LLMConfig):
    def __init__(self, model_name: str, api_type_name: str, parms: dict[str, str], settings: LLMOaiSettings):
        """

        :param api_type_name
        :param parms
        :param model_name:

        """
        super().__init__(model_name, api_type_name, parms)

        self.settings = settings

        if self.api_type_name not in ['azure', 'ollama', 'openai', 'groq', 'gemini']:
            raise ValueError(f'{__class__.__name__}: invalid api_type! {api_type_name}')
        self._api_client = None

    def __repr__(self) -> str:
        return f'[{self.__class__!s}:{self.__dict__!r}]'

    async def change_n(self, new_n: int):
        log.info(f'{self.model_name} changing n to: {new_n}')
        self.settings.n = new_n

    async def change_temp(self, new_temp: float):
        log.info(f'{self.model_name} changing temp to: {new_temp}')
        self.settings.temp = new_temp

    async def change_top_p(self, new_top_p: float):
        log.info(f'{self.model_name} changing top_p to: {new_top_p}')
        self.settings.temp = new_top_p

    async def change_max_tokens(self, new_max_tokens: int):
        log.info(f'{self.model_name} changing max_tokens to: {new_max_tokens}')
        self.settings.max_tokens = new_max_tokens

    async def change_sysmsg(self, new_system_message_name: str):
        new_system_message = data.sysmsg_all[new_system_message_name]
        log.info(f'{self.model_name} changing system message to: {new_system_message_name}:{new_system_message}')
        self.settings.system_message_name = new_system_message_name
        self.settings.system_message = new_system_message

    def _client(self) -> openai.OpenAI:
        if self._api_client is not None:
            return self._api_client

        if self.api_type_name == 'ollama':
            log.info(f'building LLM API for [{self.api_type_name}]: {self.parms.get('OLLAMA_ENDPOINT')=}')
            self._api_client = openai.OpenAI(base_url=self.parms.get('OLLAMA_ENDPOINT'),
                                             api_key='nokeyneeded')
        elif self.api_type_name == 'openai':
            log.info(f'building LLM API for [{self.api_type_name}]: {self.parms.get('OPENAI_ENDPOINT')=}, '
                     f'OPENAI_API_KEY={redact(self.parms.get('OPENAI_API_KEY'))}')
            self._api_client = openai.OpenAI(base_url=self.parms.get('OPENAI_ENDPOINT'),
                                             api_key=self.parms.get('OPENAI_API_KEY'))
        elif self.api_type_name == 'groq':
            log.info(f'building LLM API for [{self.api_type_name}]: {self.parms.get('GROQ_ENDPOINT')=}, '
                     f'GROQ_API_KEY={redact(self.parms.get('GROQ_API_KEY'))}')
            self._api_client = openai.OpenAI(base_url=self.parms.get('GROQ_OPENAI_ENDPOINT'),
                                             api_key=self.parms.get('GROQ_API_KEY'))
        elif self.api_type_name == 'gemini':
            log.info(f'building LLM API for [{self.api_type_name}]: {self.parms.get('GEMINI_ENDPOINT')=}, '
                     f'GEMINI_API_KEY={redact(self.parms.get('GEMINI_API_KEY'))}')
            self._api_client = openai.OpenAI(base_url=self.parms.get('GEMINI_OPENAI_ENDPOINT'),
                                             api_key=self.parms.get('GEMINI_API_KEY'))
        elif self.api_type_name == 'github':
            base_url = 'https://models.inference.ai.azure.com'
            log.info(f'building LLM API for [{self.api_type_name}]: {base_url=}, {redact(self.parms.get('GITHUB_TOKEN'))}')
            self._api_client = openai.OpenAI(base_url=base_url,
                                             api_key=self.parms.get('GITHUB_TOKEN'))
        elif self.api_type_name == 'azure':
            # token_provider = azure.identity.get_bearer_token_provider(
            #     azure.identity.DefaultAzureCredential(), 'https://cognitiveservices.azure.com/.default'
            # )
            # client = openai.AzureOpenAI(
            #     api_version=self.parms.get('AZURE_OPENAI_API_VERSION'),
            #     azure_endpoint=self.parms.get('AZURE_OPENAI_ENDPOINT'),
            #     azure_ad_token_provider=token_provider,
            # )
            log.info(f'building LLM API for [{self.api_type_name}]: {self.parms.get('AZURE_OPENAI_ENDPOINT')=}, '
                     f'AZURE_OPENAI_API_KEY={redact(self.parms.get('AZURE_OPENAI_API_KEY'))}, '
                     f'{self.parms.get('AZURE_OPENAI_API_VERSION')=}')
            self._api_client = openai.AzureOpenAI(azure_endpoint=self.parms.get('AZURE_OPENAI_ENDPOINT'),
                                                  api_key=self.parms.get('AZURE_OPENAI_API_KEY'),
                                                  api_version=self.parms.get('AZURE_OPENAI_API_VERSION'))
        else:
            raise ValueError(f'invalid api type! {self.api_type_name}')

        return self._api_client

    # todo: configure max_quota_retries
    def _do_chat(self, messages: list[dict], max_quota_retries: int = 10) -> LLMOaiExchange:
        # prompt is the last dict in the list
        prompt = messages[-1]['content']
        log.debug(f'{self.model_name=}, {self.settings.temp=}, {self.settings.top_p=}, {self.settings.max_tokens=}, {self.settings.n=}, '
                  f'{self.settings.system_message=} {prompt=}')

        quota_retries = 0
        while True:
            try:
                # todo: seed, etc. (by actual llm?)
                chat_completion: ChatCompletion = self._client().chat.completions.create(
                    model=self.model_name,
                    temperature=self.settings.temp,  # default 1.0, 0.0->2.0
                    top_p=self.settings.top_p,  # default 1, ~0.01->1.0
                    messages=messages,
                    max_tokens=self.settings.max_tokens,  # default 16?
                    n=self.settings.n,  # todo: openai,azure,gemini:any(?) value works; ollama: only 1 resp for any value; groq: requires 1;

                    stream=False,  # todo: allow streaming

                    # seed=27,
                    # frequency_penalty=1,  # default 0, -2.0->2.0
                    # presence_penalty=1,  # default 0, -2.0->2.0
                    # stop=[],
                )
                return LLMOaiExchange(prompt, chat_completion)
            except openai.RateLimitError as e:
                quota_retries += 1
                log.debug(f'{self.api_type_name}:{self.model_name}: rate limit exceeded attempt {quota_retries}, {("will retry" if quota_retries <= max_quota_retries else "")}')
                if quota_retries > max_quota_retries:
                    log.warning(f'chat quota error! {self.api_type_name}:{self.model_name}: rate limit exceeded all {quota_retries} retries')
                    raise e
                else:
                    time.sleep(1.0 * quota_retries)  # todo: progressive backoff?
            except (Exception,) as e:
                log.warning(f'chat error! {self.api_type_name}:{self.model_name}: {e}')
                raise e

    def chat_messages(self, messages: Iterable[tuple[str, str] | dict]) -> LLMOaiExchange:
        """
        run chat-completion from a list of messages
        :param messages: properly ordered list of either tuples of (role, value) or dicts; must include system message and prompt
        """
        # transform convo to list-of-dicts, elements are either tuples or already dicts (and I guess a mix of each, why not?)
        msgs_list = [{t[0]: t[1]} if isinstance(t, tuple) else t for t in messages]
        return self._do_chat(msgs_list)

    def chat_convo(self, convo: Iterable[LLMOaiExchange], prompt: str) -> LLMOaiExchange:
        """
        run chat-completion
        :param convo: properly ordered list of LLMOpenaiExchange's
        :param prompt: the prompt duh
        """
        messages: list[dict] = []
        if self.settings.system_message is not None and len(self.settings.system_message) > 0:
            messages.append({'role': 'system', 'content': self.settings.system_message})

        # add the convo
        for exchange in convo:
            # todo: what about previous vector-store responses?
            messages.append({'role': 'user', 'content': exchange.prompt})
            for choice in exchange.completion.choices:
                messages.append({'role': choice.message.role, 'content': choice.message.content})

        # add the prompt
        messages.append({'role': 'user', 'content': prompt})
        return self._do_chat(messages)
