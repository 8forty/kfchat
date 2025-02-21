import logging
import time

import anthropic
import dotenv
import openai
from anthropic.types import Message

import data
import logstuff
from config import redact
from llmconfig.llm_anthropic_exchange import LLMAnthropicExchange
from llmconfig.llmconfig import LLMConfig

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


class LLMAnthropicSettings:
    # todo: all of these?
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
        self.system_message = data.dummy_llm_config.sysmsg_all[init_system_message_name]


class LLMAnthropicConfig(LLMConfig):
    def __init__(self, model_name: str, provider_name: str, settings: LLMAnthropicSettings):
        """

        :param model_name:
        :param provider_name
        :param settings:

        """
        super().__init__(model_name, provider_name)

        self.settings = settings

        if self.provider_name not in list(providers_config.keys()):
            raise ValueError(f'{__class__.__name__}: invalid provider! {provider_name}')
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
        new_system_message = self.sysmsg_all[new_system_message_name]
        log.info(f'{self.model_name} changing system message to: {new_system_message_name}:{new_system_message}')
        self.settings.system_message_name = new_system_message_name
        self.settings.system_message = new_system_message

    def _client(self) -> anthropic.Anthropic:
        if self._api_client is not None:
            return self._api_client

        # todo: this is a 2nd place that lists providers :(, for now to highlight any diffs in client-setup api's
        if self.provider_name == 'anthropic':
            # endpoint = providers_config[self.provider_name]['OLLAMA_ENDPOINT']
            endpoint = 'huh?'
            key = providers_config[self.provider_name]['key']
            log.info(f'building ANTHROPIC LLM API for [{self.provider_name}]: {endpoint=} key={redact(key)}')
            self._api_client = anthropic.Anthropic(api_key=key)
        else:
            raise ValueError(f'invalid provider! {self.provider_name}')

        return self._api_client

    # todo: configure max_quota_retries
    def _do_chat(self, messages: list[dict], sysmsg: str, max_quota_retries: int = 10) -> LLMAnthropicExchange:
        # prompt is the last dict in the list
        prompt = messages[-1]['content']
        log.debug(f'{self.model_name=}, {self.settings.temp=}, {self.settings.top_p=}, {self.settings.max_tokens=}, {self.settings.n=}, '
                  f'{self.settings.system_message=} {prompt=}')

        quota_retries = 0
        retry_wait_secs = 1.0
        while True:
            try:
                # todo: seed, etc. (by actual llm?)
                chat_completion: Message = self._client().messages.create(
                    model=self.model_name,
                    temperature=self.settings.temp,  # todo: default 1.0, 0.0->1.0
                    top_p=self.settings.top_p,  # todo: default 1, ~0.01->1.0
                    messages=messages,
                    max_tokens=self.settings.max_tokens,  # default 16?
                    system=sysmsg,

                    # n=self.settings.n,  # todo: openai,azure,gemini:any(?) value works; ollama: only 1 resp for any value; groq: requires 1;

                    # stream=False,  # todo: allow streaming

                    # seed=27,
                    # frequency_penalty=1,  # default 0, -2.0->2.0
                    # presence_penalty=1,  # default 0, -2.0->2.0
                    # stop=[],
                )
                return LLMAnthropicExchange(prompt, chat_completion)
            except openai.RateLimitError as e:
                quota_retries += 1
                log.warning(f'{self.provider_name}:{self.model_name}: rate limit exceeded attempt {quota_retries}/{max_quota_retries}, '
                            f'{(f"will retry in {retry_wait_secs}s" if quota_retries <= max_quota_retries else "")}')
                if quota_retries > max_quota_retries:
                    log.warning(f'chat quota exceeded! {self.provider_name}:{self.model_name}: rate limit exceeded all {quota_retries} retries')
                    raise e
                else:
                    time.sleep(retry_wait_secs)  # todo: progressive backoff?
                    retry_wait_secs = quota_retries * quota_retries
            except (Exception,) as e:
                log.warning(f'chat error! {self.provider_name}:{self.model_name}: {e.__class__.__name__}: {e}')
                raise e
