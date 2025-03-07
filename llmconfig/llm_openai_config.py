import logging
import time
import timeit

import dotenv
import openai
from openai.types.chat import ChatCompletion

import config
import logstuff
from config import redact
from llmconfig.llm_openai_exchange import LLMOpenAIExchange
from llmconfig.llmconfig import LLMConfig, LLMSettings
from llmconfig.llmexchange import LLMMessagePair

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

dotenv.load_dotenv(override=True)
env = dotenv.dotenv_values()

providers_config = {
    'AZURE': {
        'key': env.get('kfAZURE_API_KEY'),
        'kfAZURE_API_VERSION': env.get('kfAZURE_API_VERSION'),
        'kfAZURE_ENDPOINT': env.get('kfAZURE_ENDPOINT'),
        'AZURE_AI_SEARCH_ENDPOINT': env.get('AZURE_AI_SEARCH_ENDPOINT'),
        'ai-search-api-key': env.get('AZURE_AI_SEARCH_API_KEY'),
    },
    'OLLAMA': {
        'key': 'nokeyneeded',
        'kfOLLAMA_ENDPOINT': env.get('kfOLLAMA_ENDPOINT'),
    },
    'OPENAI': {
        'key': env.get('kfOPENAI_API_KEY'),
        'kfOPENAI_CHAT_COMPLETIONS_ENDPOINT': env.get('kfOPENAI_CHAT_COMPLETIONS_ENDPOINT'),
        'kfOPENAI_ENDPOINT': env.get('kfOPENAI_ENDPOINT'),
    },
    'GROQ': {
        'key': env.get('kfGROQ_API_KEY'),
        'kfGROQ_ENDPOINT': env.get('kfGROQ_ENDPOINT'),
    },
    'GEMINI': {
        'key': env.get('kfGEMINI_API_KEY'),
        'kfGEMINI_ENDPOINT': env.get('kfGEMINI_ENDPOINT'),
    },
    'GITHUB': {
        'key': env.get('kfGITHUB_TOKEN'),
        'kfGITHUB_ENDPOINT': env.get('kfGITHUB_ENDPOINT'),
    },
    'XAI': {
        'key': env.get('kfXAI_API_KEY'),
        'kfXAI_ENDPOINT': env.get('kfXAI_ENDPOINT'),
    },
}


class LLMOpenAISettings(LLMSettings):
    def __init__(self, init_n: int, init_temp: float, init_top_p: float, init_max_tokens: int, init_system_message_name: str):
        super().__init__(init_n, init_temp, init_top_p, init_max_tokens, init_system_message_name)

    def __repr__(self) -> str:
        return f'{self.__class__!s}:{self.__dict__!r}'
        # return f'{self.__class__!s}:{self.result_id=!r},{self.metrics=!r},self.content="{self.content[0:20]}{"..." if len(self.content) > 20 else ""}"'

    @classmethod
    def from_settings(cls, rhs):
        return cls(init_n=rhs.n, init_temp=rhs.temp, init_top_p=rhs.top_p, init_max_tokens=rhs.max_tokens, init_system_message_name=rhs.system_message_name)


class LLMOpenAIConfig(LLMConfig):
    def __init__(self, model_name: str, provider: str, settings: LLMOpenAISettings):
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

    def settings(self) -> LLMSettings:
        return self._settings

    def change_settings(self, new_settings: LLMSettings) -> LLMSettings:
        old = self._settings
        self._settings = new_settings
        return old

    def copy_settings(self) -> LLMSettings:
        return LLMOpenAISettings(self._settings.n, self._settings.temp, self._settings.top_p, self._settings.max_tokens, self._settings.system_message_name)

    def _client(self) -> openai.OpenAI:
        if self._api_client is not None:
            return self._api_client

        # todo: this is a 2nd place that lists providers :(, for now to highlight any diffs in client-setup api's
        if self._provider.upper() == 'AZURE':
            # token_provider = azure.identity.get_bearer_token_provider(
            #     azure.identity.DefaultAzureCredential(), 'https://cognitiveservices.azure.com/.default'
            # )
            # client = openai.AzureOpenAI(
            #     api_version=self.parms.get('AZURE_API_VERSION'),
            #     azure_endpoint=self.parms.get('AZURE_ENDPOINT'),
            #     azure_ad_token_provider=token_provider,
            # )
            endpoint = providers_config[self._provider]['kfAZURE_ENDPOINT']
            key = providers_config[self._provider]['key']
            api_version = providers_config[self._provider]['kfAZURE_API_VERSION']
            log.info(f'building LLM API for [{self._provider}]: {endpoint=} key={redact(key)} {api_version=}')
            self._api_client = openai.AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=api_version)
        elif self._provider in config.LLMData.providers:
            endpoint = providers_config[self._provider][f'kf{self._provider.upper()}_ENDPOINT']
            key = providers_config[self._provider]['key']
            log.info(f'building LLM API for [{self._provider}]: {endpoint=} key={redact(key)}')
            self._api_client = openai.OpenAI(base_url=endpoint, api_key=key)
        # if self._provider == 'ollama':
        #     endpoint = providers_config[self._provider]['kfOLLAMA_ENDPOINT']
        #     key = providers_config[self._provider]['key']
        #     log.info(f'building LLM API for [{self._provider}]: {endpoint=} key={redact(key)}')
        #     self._api_client = openai.OpenAI(base_url=endpoint, api_key=key)
        # elif self._provider == 'openai':
        #     endpoint = providers_config[self._provider]['kfOPENAI_ENDPOINT']
        #     key = providers_config[self._provider]['key']
        #     log.info(f'building LLM API for [{self._provider}]: {endpoint=} key={redact(key)}')
        #     self._api_client = openai.OpenAI(base_url=endpoint, api_key=key)
        # elif self._provider == 'groq':
        #     endpoint = providers_config[self._provider]['kfGROQ_ENDPOINT']
        #     key = providers_config[self._provider]['key']
        #     log.info(f'building LLM API for [{self._provider}]: {endpoint=} key={redact(key)}')
        #     self._api_client = openai.OpenAI(base_url=endpoint, api_key=key)
        # elif self._provider == 'gemini':
        #     endpoint = providers_config[self._provider]['kfGEMINI_ENDPOINT']
        #     key = providers_config[self._provider]['key']
        #     log.info(f'building LLM API for [{self._provider}]: {endpoint=} key={redact(key)}')
        #     self._api_client = openai.OpenAI(base_url=endpoint, api_key=key)
        # elif self._provider == 'GITHUB':
        #     endpoint = providers_config[self._provider]['kfGITHUB_ENDPOINT']
        #     key = providers_config[self._provider]['key']
        #     log.info(f'building LLM API for [{self._provider}]: {endpoint=} key={redact(key)}')
        #     self._api_client = openai.OpenAI(base_url=endpoint, api_key=key)
        else:
            raise ValueError(f'invalid provider! {self._provider}')

        return self._api_client

    def do_chat(self, messages: list[LLMMessagePair], max_rate_limit_retries: int = 10) -> LLMOpenAIExchange:
        # prompt is the last dict in the list
        prompt = messages[-1].content

        rate_limit_retries = 0
        retry_wait_secs = 1.0
        while True:
            try:
                messages_list: list[dict] = [{'role': 'system', 'content': self._settings.system_message}]
                messages_list.extend([{'role': pair.role, 'content': pair.content} for pair in messages])
                log.debug(f'{self.model_name} n:{self._settings.n} temp:{self._settings.temp} top_p:{self._settings.top_p}, max_tok:{self._settings.max_tokens} prompt:"{prompt}" msgs:{messages_list}')

                start = timeit.default_timer()
                chat_completion: ChatCompletion = self._client().chat.completions.create(
                    model=self.model_name,
                    temperature=self._settings.temp,  # default 1.0, 0.0->2.0
                    top_p=self._settings.top_p,  # default 1, ~0.01->1.0
                    messages=messages_list,
                    max_tokens=self._settings.max_tokens,  # default 16?
                    n=self._settings.n,

                    stream=False,

                    seed=27,
                    # frequency_penalty=1,  # default 0, -2.0->2.0
                    # presence_penalty=1,  # default 0, -2.0->2.0
                    # stop=[],
                )
                return LLMOpenAIExchange(prompt, chat_completion=chat_completion, model_name=self.model_name, provider=self._provider,
                                         response_duration_seconds=timeit.default_timer() - start, settings=self._settings)
            except openai.RateLimitError as e:
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
