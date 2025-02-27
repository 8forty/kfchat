import logging
import time
import timeit

import anthropic
import dotenv
from anthropic.types import Message

import logstuff
from config import redact
from llmconfig.llm_anthropic_exchange import LLMAnthropicExchange
from llmconfig.llmconfig import LLMConfig, LLMSettings
from llmconfig.llmexchange import LLMMessagePair

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

    def __repr__(self) -> str:
        return f'{self.__class__!s}:{self.__dict__!r}'
        # return f'{self.__class__!s}:{self.result_id=!r},{self.metrics=!r},self.content="{self.content[0:20]}{"..." if len(self.content) > 20 else ""}"'

    @classmethod
    def from_settings(cls, rhs):
        return cls(init_n=rhs.n, init_temp=rhs.temp, init_top_p=rhs.top_p, init_max_tokens=rhs.max_tokens, init_system_message_name=rhs.system_message_name)

    def numbers_oneline_logging_str(self) -> str:
        return f'n?:{self.n},temp:{self.temp},top_p:{self.top_p},max_tokens:{self.max_tokens}'

    def texts_oneline_logging_str(self) -> str:
        return f'sysmsg:{self.system_message}'


class LLMAnthropicConfig(LLMConfig):
    def __init__(self, model_name: str, provider: str, settings: LLMAnthropicSettings):
        """

        :param model_name:
        :param provider
        :param settings:

        """
        super().__init__(model_name, provider)

        self._settings = settings

        if self._provider not in list(providers_config.keys()):
            raise ValueError(f'{__class__.__name__}: invalid provider! {provider}')
        self._api_client = None

    def __repr__(self) -> str:
        return f'{self.__class__!s}:{self.__dict__!r}'

    def change_settings(self, new_settings: LLMSettings) -> LLMSettings:
        old = self._settings
        self._settings = new_settings
        return old

    def settings(self) -> LLMSettings:
        return self._settings

    def copy_settings(self) -> LLMSettings:
        return LLMAnthropicSettings(self._settings.n, self._settings.temp, self._settings.top_p, self._settings.max_tokens, self._settings.system_message_name)

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

        quota_retries = 0
        retry_wait_secs = 1.0
        while True:
            try:
                messages_list: list[dict] = [{'role': pair.role, 'content': pair.content} for pair in messages]
                log.debug(f'{self.model_name} n:{self._settings.n} temp:{self._settings.temp} top_p:{self._settings.top_p}, max_tok:{self._settings.max_tokens} prompt:"{prompt}" msgs:{messages_list}')

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
