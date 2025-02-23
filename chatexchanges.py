import logging

import logstuff
from llmconfig.llmexchange import LLMExchange

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


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

    def __init__(self, prompt: str, response_duration_secs: float, source: str, mode: str,
                 llm_exchange: LLMExchange | None, vector_store_response: VectorStoreResponse | None):
        self.prompt: str = prompt
        self.response_duration_secs: float = response_duration_secs
        self.source: str = source
        self.mode: str = mode

        self.llm_exchange = llm_exchange
        self.vector_store_response: VectorStoreResponse | None = vector_store_response

        self._overflowed = False

    def problems(self) -> dict[int, str]:
        """
        :return: dict of [response-idx -> text of problem]
        """
        return self.llm_exchange.problems

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

    def clear(self):
        self._exchanges.clear()

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
