import logging
import time
import timeit
from typing import Iterable

import anthropic
import dotenv
import openai
from anthropic.types import Message

import data
import logstuff
from config import redact
from llmconfig.llm_anthropic_exchange import LLMAnthropicExchange
from llmconfig.llmconfig import LLMConfig, LLMSettings
from llmconfig.llmexchange import LLMExchange, LLMMessagePair

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

dotenv.load_dotenv(override=True)
env = dotenv.dotenv_values()

providers_config = {
    'anthropic': {
        'key': env.get('kfANTHROPIC_TOKEN'),
        'ANTHROPIC_API_VERSION': env.get('ANTHROPIC_API_VERSION'),
        'ANTHROPIC_ENDPOINT': env.get('ANTHROPIC_ENDPOINT'),
    },
}


class LLMAnthropicSettings(LLMSettings):
    # todo: doesn't support n
    def __init__(self, init_n: int, init_temp: float, init_top_p: float, init_max_tokens: int, init_system_message_name: str):
        super().__init__(init_n, init_temp, init_top_p, init_max_tokens, init_system_message_name)
        self.system_message = LLMConfig.sysmsg_all[init_system_message_name]

    def numbers_oneline_logging_str(self) -> str:
        return f'temp:{self.temp},top_p:{self.top_p},max_tokens:{self.max_tokens}'

    def texts_oneline_logging_str(self) -> str:
        return f'sysmsg:{self.system_message}'


class LLMAnthropicConfig(LLMConfig):
    def __init__(self, model_name: str, provider_name: str, settings: LLMAnthropicSettings):
        """

        :param model_name:
        :param provider_name
        :param settings:

        """
        super().__init__(model_name, provider_name)

        self._settings = settings

        if self._provider not in list(providers_config.keys()):
            raise ValueError(f'{__class__.__name__}: invalid provider! {provider_name}')
        self._api_client = None

    def __repr__(self) -> str:
        return f'[{self.__class__!s}:{self.__dict__!r}]'

    def settings(self) -> LLMSettings:
        return self._settings

    def copy_settings(self) -> LLMSettings:
        return LLMAnthropicSettings(self._settings.n, self._settings.temp, self._settings.top_p, self._settings.max_tokens, self._settings.system_message_name)

    async def change_n(self, new_n: int):
        log.info(f'{self.model_name} changing n to: {new_n}')
        self._settings.n = new_n

    async def change_temp(self, new_temp: float):
        log.info(f'{self.model_name} changing temp to: {new_temp}')
        self._settings.temp = new_temp

    async def change_top_p(self, new_top_p: float):
        log.info(f'{self.model_name} changing top_p to: {new_top_p}')
        self._settings.temp = new_top_p

    async def change_max_tokens(self, new_max_tokens: int):
        log.info(f'{self.model_name} changing max_tokens to: {new_max_tokens}')
        self._settings.max_tokens = new_max_tokens

    async def change_sysmsg(self, new_system_message_name: str):
        new_system_message = self.sysmsg_all[new_system_message_name]
        log.info(f'{self.model_name} changing system message to: {new_system_message_name}:{new_system_message}')
        self._settings.system_message_name = new_system_message_name
        self._settings.system_message = new_system_message

    def _client(self) -> anthropic.Anthropic:
        if self._api_client is not None:
            return self._api_client

        # todo: this is a 2nd place that lists providers :(, for now to highlight any diffs in client-setup api's
        if self._provider == 'anthropic':
            # endpoint = providers_config[self.provider_name]['OLLAMA_ENDPOINT']
            endpoint = 'huh?'
            key = providers_config[self._provider]['key']
            log.info(f'building ANTHROPIC LLM API for [{self._provider}]: {endpoint=} key={redact(key)}')
            self._api_client = anthropic.Anthropic(api_key=key)
        else:
            raise ValueError(f'invalid provider! {self._provider}')

        return self._api_client

    # todo: configure max_quota_retries
    def do_chat(self, messages: list[LLMMessagePair], max_quota_retries: int = 10) -> LLMAnthropicExchange:
        # todo: this is clumsy
        # prompt is the last dict in the list
        prompt = messages[-1].content
        log.debug(f'{self.model_name=}, {self._settings.temp=}, {self._settings.top_p=}, {self._settings.max_tokens=}, {self._settings.n=}, '
                  f'{self._settings.system_message=} {prompt=}')

        quota_retries = 0
        retry_wait_secs = 1.0
        while True:
            try:
                messages_list: list[dict] = [{'role': pair.role, 'content': pair.content} for pair in messages]
                start = timeit.default_timer()
                # todo: seed, etc. (by actual llm?)
                message: Message = self._client().messages.create(
                    model=self.model_name,
                    temperature=self._settings.temp,  # todo: default 1.0, 0.0->1.0
                    top_p=self._settings.top_p,  # todo: default 1, ~0.01->1.0
                    messages=messages_list,
                    max_tokens=self._settings.max_tokens,  # default 16?
                    system=self._settings.system_message

                    # n=self._settings.n,  # todo: not in anthropic's api

                    # stream=False,  # todo: allow streaming

                    # seed=27,
                    # frequency_penalty=1,  # default 0, -2.0->2.0
                    # presence_penalty=1,  # default 0, -2.0->2.0
                    # stop=[],
                )
                return LLMAnthropicExchange(prompt=prompt, message=message, provider=self._provider, model_name=self.model_name,
                                            settings=self._settings, response_duration_seconds=timeit.default_timer() - start)
            except anthropic.RateLimitError as e:
                quota_retries += 1
                log.warning(f'{self._provider}:{self.model_name}: rate limit exceeded attempt {quota_retries}/{max_quota_retries}, '
                            f'{(f"will retry in {retry_wait_secs}s" if quota_retries <= max_quota_retries else "")}')
                if quota_retries > max_quota_retries:
                    log.warning(f'chat quota exceeded! {self._provider}:{self.model_name}: rate limit exceeded all {quota_retries} retries')
                    raise e
                else:
                    time.sleep(retry_wait_secs)  # todo: progressive backoff?
                    retry_wait_secs = quota_retries * quota_retries
            except (Exception,) as e:
                log.warning(f'chat error! {self._provider}:{self.model_name}: {e.__class__.__name__}: {e}')
                raise e
