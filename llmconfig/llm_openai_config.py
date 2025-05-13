import logging
import time
import timeit

import dotenv
import openai
from openai.types.chat import ChatCompletion

import config
import logstuff
from basesettings import BaseSettings
from config import redact
from llmconfig.llm_openai_exchange import LLMOpenAIExchange
from llmconfig.llmconfig import LLMConfig, LLMSettings
from llmconfig.llmexchange import LLMMessagePair

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

dotenv.load_dotenv(override=True)
env = dotenv.dotenv_values()

providers_config = {
    'LLAMACPP': {
        'key': 'dont need one',
        'kfLLAMACPP_CHAT_COMPLETIONS_ENDPOINT': 'http://localhost:27272/v1/chat/completions',
        'kfLLAMACPP_ENDPOINT': 'http://localhost:27272/v1',
    },
    'AZURE': {
        'key': env.get('kfAZURE_API_KEY'),
        'kfAZURE_API_VERSION': env.get('kfAZURE_API_VERSION'),
        'kfAZURE_ENDPOINT': env.get('kfAZURE_ENDPOINT'),
        'AZURE_AI_SEARCH_ENDPOINT': env.get('AZURE_AI_SEARCH_ENDPOINT'),
        'ai-search-api-key': env.get('AZURE_AI_SEARCH_API_KEY'),
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
    'CEREBRAS': {
        'key': env.get('kfCEREBRAS_API_KEY'),
        'kfCEREBRAS_ENDPOINT': env.get('kfCEREBRAS_ENDPOINT'),
    },
    'GEMINI': {
        'key': env.get('kfGEMINI_API_KEY'),
        'kfGEMINI_ENDPOINT': env.get('kfGEMINI_ENDPOINTv1beta'),
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
        super().__init__()
        self.n = init_n
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
        return cls(init_n=rhs.n, init_temp=rhs.temp, init_top_p=rhs.top_p, init_max_tokens=rhs.max_tokens, init_system_message_name=rhs.system_message_name)

    def numbers_oneline_logging_str(self) -> str:
        return f'n:{self.n},temp:{self.temp},top_p:{self.top_p},max_tokens:{self.max_tokens}'

    def texts_oneline_logging_str(self) -> str:
        return f'sysmsg:{self.system_message}'

    def specs(self) -> list[BaseSettings.SettingsSpec]:
        sysmsg_names = [key for key in config.LLMData.sysmsg_all]
        return [
            BaseSettings.SettingsSpec(label='n', options=[i for i in range(1, 10)], value=self.n, tooltip='number of results per query'),
            BaseSettings.SettingsSpec(label='temp', options=[float(t) / 10.0 for t in range(0, 21)], value=self.temp, tooltip='responses: 0=very predictable, 2=very random/creative'),
            BaseSettings.SettingsSpec(label='top_p', options=[float(t) / 10.0 for t in range(0, 11)], value=self.top_p, tooltip='responses: 0=less random, 1 more random'),
            BaseSettings.SettingsSpec(label='max_tokens', options=[80, 200, 400, 800, 1000, 1500, 2000], value=self.max_tokens, tooltip='max tokens in response'),
            BaseSettings.SettingsSpec(label='system_message_name', options=sysmsg_names, value=self.system_message_name, tooltip='system/setup text sent with each prompt'),
        ]

    async def change(self, label: str, value: any) -> None:
        if label == 'n':
            self.n = value
        elif label == 'temp':
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
        #     endpoint = providers_config[self._provider]['kfGEMINI_ENDPOINTv1beta']
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

    def generate_chat_completion(self, messages: list[dict]) -> ChatCompletion:

        chat_completion: ChatCompletion = self._client().chat.completions.create(
            model=self.model_name,
            temperature=self._settings.temp,  # default 1.0, 0.0->2.0
            top_p=self._settings.top_p,  # default 1, ~0.01->1.0
            messages=messages,
            max_tokens=self._settings.max_tokens,  # default 16?
            n=self._settings.n,

            # stream=False,

            seed=27,

            # frequency_penalty=1,  # default 0, -2.0->2.0
            # presence_penalty=1,  # default 0, -2.0->2.0
            # stop=[],
        )
        return chat_completion

    def _chat(self, messages: list[LLMMessagePair], context: list[str] | None, max_rate_limit_retries: int = 10) -> LLMOpenAIExchange:
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
                messages_list: list[dict] = [{'role': 'system', 'content': sysmsg}]
                messages_list.extend([{'role': pair.role, 'content': pair.content} for pair in messages])
                log.debug(f'{self._provider}.{self.model_name} n:{self._settings.n} temp:{self._settings.temp} top_p:{self._settings.top_p}, max_tok:{self._settings.max_tokens} prompt:"{prompt}" msgs:{messages_list}')

                start = timeit.default_timer()
                chat_completion: ChatCompletion = self.generate_chat_completion(messages_list)
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
