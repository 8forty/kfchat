import logging

from openai.types.chat import ChatCompletion

import logstuff
from llmconfig import LLMConfig

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMResponse:
    def __init__(self, chat_completion: ChatCompletion, llm_config: LLMConfig):
        self.chat_completion: ChatCompletion = chat_completion
        self.api_type: str = llm_config.llmapi.type()
        self.model_name: str = llm_config.model_name
        self.n: int = llm_config.n
        self.temp: float = llm_config.temp
        self.top_p: float = llm_config.top_p
        self.max_tokens: int = llm_config.max_tokens

    def __repr__(self) -> str:
        return f'[{self.__class__!s}:{self.__dict__!r}]'


class VectorStoreResult:
    def __init__(self, result_id: str, metrics: dict, content: str):
        self.result_id: str = result_id
        self.metrics: dict = metrics
        self.content: str = content

    def __repr__(self) -> str:
        return f'[{self.__class__!s}:{self.__dict__!r}]'
        # return f'[{self.__class__!s}:{self.result_id=!r},{self.metrics=!r},self.content="{self.content[0:20]}{"..." if len(self.content) > 20 else ""}"]'


class VectorStoreResponse:
    def __init__(self, results: list[VectorStoreResult]):
        self.results = results

    def __repr__(self) -> str:
        return self.results.__repr__()


class ChatExchange:
    # ChatCompletion (full openai version):
    #     id: str  A unique identifier for the chat completion
    #     choices: List[Choice]  A list of chat completion choices. Can be more than one if `n` is greater than 1.
    #     created: int  The Unix timestamp (in seconds) of when the chat completion was created.
    #     model: str  The model used for the chat completion.
    #     object: Literal["chat.completion"]  The object type, which is always `chat.completion`
    #     service_tier: Optional[Literal["scale", "default"]] = None  The service tier used, included iff the `service_tier` parameter is specified in the request.
    #     system_fingerprint: Optional[str] = None  This fingerprint represents the backend configuration that the model runs with, in conjunction with the `seed` to detect backend changes
    #     usage: Optional[CompletionUsage] = None  Usage statistics for the completion request.

    def __init__(self, prompt: str, response_duration_secs: float,
                 llm_response: LLMResponse | None, vector_store_response: VectorStoreResponse | None):
        self.prompt: str = prompt
        self.response_duration_secs: float = response_duration_secs

        self.llm_response: LLMResponse | None = llm_response
        self.vector_store_response: VectorStoreResponse | None = vector_store_response

        self._stop_problems: dict[int, str] = {}
        self._overflowed = False

        # calc stop_problems if there's an llm response
        if self.llm_response is not None:
            for i in range(0, len(llm_response.chat_completion.choices)):
                choice = llm_response.chat_completion.choices[i]
                stop_problem = ''
                match choice.finish_reason:
                    case 'length':
                        stop_problem = 'too many tokens'
                    case 'content_filter':
                        stop_problem = 'flagged by content filter'
                    case 'tool_calls':
                        stop_problem = 'called a tool'
                    case 'function_call':
                        stop_problem = 'called a function'

                if len(stop_problem) > 0:
                    self._stop_problems[i] = stop_problem  # add problem to the dict

    def stop_problems(self) -> dict[int, str]:
        """
        get stop problems reported in ChatCompletion
        :return: dict of choice-idx -> text of stop problem
        """
        return self._stop_problems

    def overflowed(self) -> bool:
        return self._overflowed

    def __repr__(self) -> str:
        return f'[{self.__class__!s}:{self.__dict__!r}]'


class ChatExchanges:
    _next_id: int = 1

    def __init__(self, max_exchanges: int):
        """
        initilizes a list to hold at most max_exchanges ChatExchange objects with a fresh id
        :param max_exchanges: when this number of ChatExchange objects is exceeded, oldest is removed
        """
        super().__init__()
        """ a 0-or-more question prompt and list of ChatResponse objects, one for each question """
        self._exchanges: list[ChatExchange] = []

        self._max_exchanges = max_exchanges
        self._id = ChatExchanges._next_id
        ChatExchanges._next_id += 1

    def id(self) -> int:
        return self._id

    def empty(self) -> bool:
        return len(self._exchanges) == 0

    def list(self) -> list[ChatExchange]:
        return self._exchanges

    def len(self) -> int:
        return len(self._exchanges)

    def max_exchanges(self) -> int:
        return self._max_exchanges

    def append(self, ce: ChatExchange):
        self._exchanges.append(ce)
        if len(self._exchanges) > self._max_exchanges:
            log.debug(f'reducing overflow id:{self._id}: {len(self._exchanges)} > {self._max_exchanges}')
            self._exchanges.pop(0)
            ce._overflowed = True

    def __repr__(self) -> str:
        return self._exchanges.__repr__()
