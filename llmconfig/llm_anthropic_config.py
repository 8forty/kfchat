import logging
import time
import timeit

import anthropic
import dotenv
from anthropic.types import Message
from openai.types.chat import ChatCompletion

import config
import logstuff
from basesettings import BaseSettings
from config import redact
from llmconfig.llm_anthropic_exchange import LLMAnthropicExchange
from llmconfig.llmconfig import LLMConfig, LLMSettings
from llmconfig.llmexchange import LLMMessagePair

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

dotenv.load_dotenv(override=True)
env = dotenv.dotenv_values()

providers_config = {
    'ANTHROPIC': {
        'key': env.get('kfANTHROPIC_TOKEN'),
        'ANTHROPIC_API_VERSION': env.get('kfANTHROPIC_API_VERSION'),
        'ANTHROPIC_ENDPOINT': "doesn't need one!",
    },
}


class LLMAnthropicSettings(LLMSettings):

    def __init__(self, init_temp: float, init_top_p: float, init_max_tokens: int, init_system_message_name: str):
        super().__init__()
        self.temp = init_temp
        self.top_p = init_top_p
        self.max_tokens = init_max_tokens
        self.system_message_name = init_system_message_name
        self.system_message = config.LLMData.sysmsg_all[init_system_message_name]

    def __repr__(self) -> str:
        return f'{self.__class__!s}:{self.__dict__!r}'
        # return f'{self.__class__!s}:{self.result_id=!r},{self.metrics=!r},self.content="{self.content[0:20]}{"..." if len(self.content) > 20 else ""}"'

    @classmethod
    def from_settings(cls, rhs):
        return cls(init_temp=rhs.temp, init_top_p=rhs.top_p, init_max_tokens=rhs.max_tokens, init_system_message_name=rhs.system_message_name)

    def numbers_oneline_logging_str(self) -> str:
        return f'temp:{self.temp},top_p:{self.top_p},max_tokens:{self.max_tokens}'

    def texts_oneline_logging_str(self) -> str:
        return f'sysmsg:{self.system_message}'

    def specs(self) -> list[BaseSettings.SettingsSpec]:
        sysmsg_names = [key for key in config.LLMData.sysmsg_all]
        return [
            BaseSettings.SettingsSpec(label='temp', options=[float(t) / 10.0 for t in range(0, 21)], value=self.temp, tooltip='responses: 0=very predictable, 2=very random/creative'),
            BaseSettings.SettingsSpec(label='top_p', options=[float(t) / 10.0 for t in range(0, 11)], value=self.top_p, tooltip='responses: 0=less random, 1 more random'),
            BaseSettings.SettingsSpec(label='max_tokens', options=[80, 200, 400, 800, 1000, 1500, 2000], value=self.max_tokens, tooltip='max tokens in response'),
            BaseSettings.SettingsSpec(label='system_message_name', options=sysmsg_names, value=self.system_message_name, tooltip='system/setup text sent with each prompt'),
        ]

    async def change(self, label: str, value: any) -> None:
        if label == 'temp':
            self.temp = value
        elif label == 'top_p':
            self.top_p = value
        elif label == 'max_tokens':
            self.max_tokens = value
        elif label == 'system_message_name':
            self.system_message_name = value
            self.system_message = config.LLMData.sysmsg_all[value]
        else:
            raise ValueError(f'bad label! {label}')


class LLMAnthropicConfig(LLMConfig):
    def __init__(self, model_name: str, provider: str, settings: LLMAnthropicSettings):
        """

        :param model_name:
        :param provider
        :param settings:

        """
        super().__init__(model_name, provider)

        self._settings = settings

        # todo: this is duplicated in openai config
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

        if self._provider == 'ANTHROPIC':
            # endpoint = providers_config[self.provider_name]['OLLAMA_ENDPOINT']
            endpoint = "doesn't need one!"
            key = providers_config[self._provider]['key']
            log.info(f'building ANTHROPIC LLM API for [{self._provider}]: {endpoint=} key={redact(key)}')
            self._api_client = anthropic.Anthropic(api_key=key)
        else:
            raise ValueError(f'invalid provider! {self._provider}')

        return self._api_client

    def _chat(self, messages: list[LLMMessagePair], context: list[str] | None, max_rate_limit_retries: int = 10) -> LLMAnthropicExchange:
        # prompt is the last dict in the list by openai's convention
        # todo: this is clumsy
        prompt = messages[-1].content

        # normal or RAG?
        if context is None:
            sysmsg = self._settings.system_message
        else:
            sysmsg = config.LLMData.rag1_sysmsg.format(sysmsg=self._settings.system_message, context=context)

        rate_limit_retries = 0
        retry_wait_secs = 1.0
        while True:
            try:
                messages_list: list[dict] = [{'role': pair.role, 'content': pair.content} for pair in messages]
                log.debug(f'{self._provider}.{self.model_name} temp:{self._settings.temp} top_p:{self._settings.top_p}, max_tok:{self._settings.max_tokens} prompt:"{prompt}" msgs:{messages_list}')

                start = timeit.default_timer()
                message: Message = self._client().messages.create(
                    model=self.model_name,
                    temperature=self._settings.temp,  # default 1.0, 0.0->1.0
                    top_p=self._settings.top_p,  # default 1, ~0.01->1.0
                    messages=messages_list,
                    max_tokens=self._settings.max_tokens,  # default 16?
                    system=sysmsg,

                    # n=self._settings.n,  # not in anthropic's api

                    # stream=False,  # todo: allow streaming

                    # seed=27,
                    # frequency_penalty=1,  # default 0, -2.0->2.0
                    # presence_penalty=1,  # default 0, -2.0->2.0
                    # stop=[],
                )
                return LLMAnthropicExchange(prompt=prompt, message=message, provider=self._provider, model_name=self.model_name,
                                            settings=self._settings, response_duration_seconds=timeit.default_timer() - start)
            except anthropic.RateLimitError as e:
                rate_limit_retries += 1
                log.warning(f'{self._provider}:{self.model_name}: rate limit exceeded attempt {rate_limit_retries}/{max_rate_limit_retries}, '
                            f'{(f"will retry in {retry_wait_secs}s" if rate_limit_retries <= max_rate_limit_retries else "")}')
                if rate_limit_retries > max_rate_limit_retries:
                    log.warning(f'chat {self._provider}:{self.model_name}: rate limit exceeded, all {rate_limit_retries} retries failed')
                    raise e
                else:
                    time.sleep(retry_wait_secs)
                    retry_wait_secs = rate_limit_retries * rate_limit_retries
            except (Exception,) as e:
                log.warning(f'chat error! {self._provider}:{self.model_name}: {e.__class__.__name__}: {e}')
                raise e
