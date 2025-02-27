import logging
import time
import timeit

import dotenv
import openai
from openai.types.chat import ChatCompletion

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
    'azure': {
        'key': env.get('kfAZURE_OPENAI_API_KEY'),
        'AZURE_OPENAI_API_VERSION': env.get('AZURE_OPENAI_API_VERSION'),
        'AZURE_OPENAI_ENDPOINT': env.get('AZURE_OPENAI_ENDPOINT'),
        'AZURE_AI_SEARCH_ENDPOINT': env.get('AZURE_AI_SEARCH_ENDPOINT'),
        'ai-search-api-key': env.get('AZURE_AI_SEARCH_API_KEY'),
    },
    'ollama': {
        'key': 'nokeyneeded',
        'OLLAMA_ENDPOINT': 'http://localhost:11434/v1/',
    },
    'openai': {
        'key': env.get('kfOPENAI_API_KEY'),
        'OPENAI_CHAT_COMPLETIONS_ENDPOINT': env.get('OPENAI_CHAT_COMPLETIONS_ENDPOINT'),
        'OPENAI_ENDPOINT': env.get('OPENAI_ENDPOINT'),
    },
    'groq': {
        'key': env.get('kfGROQ_API_KEY'),
        'GROQ_OPENAI_ENDPOINT': env.get('GROQ_OPENAI_ENDPOINT'),
    },
    'gemini': {
        'key': env.get('kfGEMINI_API_KEY'),
        'GEMINI_OPENAI_ENDPOINT': env.get('GEMINI_OPENAI_ENDPOINT'),
    },
    'github': {
        'key': env.get('kfGITHUB_TOKEN'),
        'GITHUB_ENDPOINT': env.get('GITHUB_ENDPOINT'),
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

        # todo: this is a 2nd place that lists api types :(, for now to highlight any diffs in client-setup api's
        if self._provider == 'ollama':
            endpoint = providers_config[self._provider]['OLLAMA_ENDPOINT']
            key = providers_config[self._provider]['key']
            log.info(f'building LLM API for [{self._provider}]: {endpoint=} key={redact(key)}')
            self._api_client = openai.OpenAI(base_url=endpoint, api_key=key)
        elif self._provider == 'openai':
            endpoint = providers_config[self._provider]['OPENAI_ENDPOINT']
            key = providers_config[self._provider]['key']
            log.info(f'building LLM API for [{self._provider}]: {endpoint=} key={redact(key)}')
            self._api_client = openai.OpenAI(base_url=endpoint, api_key=key)
        elif self._provider == 'groq':
            endpoint = providers_config[self._provider]['GROQ_OPENAI_ENDPOINT']
            key = providers_config[self._provider]['key']
            log.info(f'building LLM API for [{self._provider}]: {endpoint=} key={redact(key)}')
            self._api_client = openai.OpenAI(base_url=endpoint, api_key=key)
        elif self._provider == 'gemini':
            endpoint = providers_config[self._provider]['GEMINI_OPENAI_ENDPOINT']
            key = providers_config[self._provider]['key']
            log.info(f'building LLM API for [{self._provider}]: {endpoint=} key={redact(key)}')
            self._api_client = openai.OpenAI(base_url=endpoint, api_key=key)
        elif self._provider == 'github':
            endpoint = providers_config[self._provider]['GITHUB_ENDPOINT']
            key = providers_config[self._provider]['key']
            log.info(f'building LLM API for [{self._provider}]: {endpoint=} key={redact(key)}')
            self._api_client = openai.OpenAI(base_url=endpoint, api_key=key)
        elif self._provider == 'azure':
            # token_provider = azure.identity.get_bearer_token_provider(
            #     azure.identity.DefaultAzureCredential(), 'https://cognitiveservices.azure.com/.default'
            # )
            # client = openai.AzureOpenAI(
            #     api_version=self.parms.get('AZURE_OPENAI_API_VERSION'),
            #     azure_endpoint=self.parms.get('AZURE_OPENAI_ENDPOINT'),
            #     azure_ad_token_provider=token_provider,
            # )
            endpoint = providers_config[self._provider]['AZURE_OPENAI_ENDPOINT']
            key = providers_config[self._provider]['key']
            api_version = providers_config[self._provider]['AZURE_OPENAI_API_VERSION']
            log.info(f'building LLM API for [{self._provider}]: {endpoint=} key={redact(key)} {api_version=}')
            self._api_client = openai.AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=api_version)
        else:
            raise ValueError(f'invalid api type! {self._provider}')

        return self._api_client

    # todo: configure max_quota_retries
    def do_chat(self, messages: list[LLMMessagePair], max_quota_retries: int = 10) -> LLMOpenAIExchange:
        # prompt is the last dict in the list
        prompt = messages[-1].content

        quota_retries = 0
        retry_wait_secs = 1.0
        while True:
            try:
                messages_list: list[dict] = [{'role': 'system', 'content': self._settings.system_message}]
                messages_list.extend([{'role': pair.role, 'content': pair.content} for pair in messages])
                log.debug(f'{self.model_name} n:{self._settings.n} temp:{self._settings.temp} top_p:{self._settings.top_p}, max_tok:{self._settings.max_tokens} prompt:"{prompt}" msgs:{messages_list}')

                start = timeit.default_timer()
                # todo: seed, etc. (by actual llm?)
                chat_completion: ChatCompletion = self._client().chat.completions.create(
                    model=self.model_name,
                    temperature=self._settings.temp,  # default 1.0, 0.0->2.0
                    top_p=self._settings.top_p,  # default 1, ~0.01->1.0
                    messages=messages_list,
                    max_tokens=self._settings.max_tokens,  # default 16?
                    n=self._settings.n,  # todo: openai,azure,gemini:any(?) value works; ollama: only 1 resp for any value; groq: requires 1;

                    stream=False,  # todo: allow streaming

                    # seed=27,
                    # frequency_penalty=1,  # default 0, -2.0->2.0
                    # presence_penalty=1,  # default 0, -2.0->2.0
                    # stop=[],
                )
                return LLMOpenAIExchange(prompt, chat_completion=chat_completion, model_name=self.model_name, provider=self._provider,
                                         response_duration_seconds=timeit.default_timer() - start, settings=self._settings)
            except openai.RateLimitError as e:
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
