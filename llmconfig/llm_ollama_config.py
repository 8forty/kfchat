import logging
import timeit

import dotenv
import ollama
from ollama import ChatResponse

import config
import logstuff
from basesettings import BaseSettings
from llmconfig.llm_ollama_exchange import LLMOllamaExchange
from llmconfig.llmconfig import LLMConfig, LLMSettings
from llmconfig.llmexchange import LLMMessagePair

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

dotenv.load_dotenv(override=True)
env = dotenv.dotenv_values()

providers_config = {
    'OLLAMA': {
        'key': env.get('kfOLLAMA_API_KEY'),
        'OLLAMA_ENDPOINT': env.get('kfOLLAMA_ENDPOINT'),
    },
}


class LLMOllamaSettings(LLMSettings):

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
        return cls(init_n=rhs.n, init_temp=rhs.temp, init_top_p=rhs.top_p, init_max_tokens=rhs.max_tokens,
                   init_system_message_name=rhs.system_message_name)

    def numbers_oneline_logging_str(self) -> str:
        return f'n:{self.n},temp:{self.temp},top_p:{self.top_p},max_tokens:{self.max_tokens}'

    def texts_oneline_logging_str(self) -> str:
        return f'sysmsg:{self.system_message}'

    def specs(self) -> list[BaseSettings.SettingsSpec]:
        sysmsg_names = [key for key in config.LLMData.sysmsg_all]
        return [
            BaseSettings.SettingsSpec(label='n', options=[i for i in range(1, 10)], value=self.n,
                                      tooltip='number of results per query'),
            BaseSettings.SettingsSpec(label='temp', options=[float(t) / 10.0 for t in range(0, 21)], value=self.temp,
                                      tooltip='responses: 0=very predictable, 2=very random/creative'),
            BaseSettings.SettingsSpec(label='top_p', options=[float(t) / 10.0 for t in range(0, 11)], value=self.top_p,
                                      tooltip='responses: 0=less random, 1 more random'),
            BaseSettings.SettingsSpec(label='max_tokens', options=[80, 200, 400, 800, 1000, 1500, 2000], value=self.max_tokens,
                                      tooltip='max tokens in response'),
            BaseSettings.SettingsSpec(label='system_message_name', options=sysmsg_names, value=self.system_message_name,
                                      tooltip='system/setup text sent with each prompt'),
        ]

    async def change(self, label: str, value: any) -> None:
        if label == 'n':
            self.n = value
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


class LLMOllamaConfig(LLMConfig):
    def __init__(self, model_name: str, provider: str, settings: LLMOllamaSettings):
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
        return LLMOllamaSettings(self._settings.n, self._settings.temp, self._settings.top_p, self._settings.max_tokens,
                                 self._settings.system_message_name)

    def _client(self) -> ollama.Client:
        if self._api_client is not None:
            return self._api_client

        if self._provider == 'OLLAMA':
            endpoint = providers_config[self._provider]['OLLAMA_ENDPOINT']
            # key = providers_config[self._provider]['key']
            log.info(f'building OLLAMA LLM API for [{self._provider}]: {endpoint=} key=(no key)')
            self._api_client = ollama.Client(host=endpoint)
        else:
            raise ValueError(f'invalid provider! {self._provider}')

        return self._api_client

    def _chat(self, messages: list[LLMMessagePair], context: list[str] | None, max_rate_limit_retries: int = 10) \
            -> LLMOllamaExchange:
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
                log.debug(f'{self._provider}.{self.model_name} n:{self._settings.n} temp:{self._settings.temp} '
                          f'top_p:{self._settings.top_p}, max_tok:{self._settings.max_tokens} prompt:"{prompt}" '
                          f'msgs:{messages_list}')

                start = timeit.default_timer()
                chat_response: ChatResponse = self._client().chat(
                    model=self.model_name,
                    messages=messages_list,
                    stream=False,  # todo: allow streaming

                    # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
                    options={
                        'num_predict': self._settings.max_tokens,  # default -1 (infinite)
                        # 'seed': 27,  # default 0
                        'temperature': self._settings.temp,  # default 0.8
                        'top_p': self._settings.top_p,  # default 0.9
                        # 'min_p': , # default 0.0
                        # 'top_k': ,  # default 40
                        # 'num_ctx': 0,  # default 2048
                        # 'repeat_last_n': , # default 64, 0=disabled, -1=num_ctx
                        # 'repeat_penalty': , # default 1.1
                        # 'stop': [],
                        # mirostat*
                    },
                )
                return LLMOllamaExchange(prompt=prompt, chat_response=chat_response, provider=self._provider,
                                         model_name=self.model_name, settings=self._settings,
                                         response_duration_seconds=timeit.default_timer() - start)
            except (Exception,) as e:
                log.warning(f'chat error! {self._provider}:{self.model_name}: {e.__class__.__name__}: {e}')
                raise e
