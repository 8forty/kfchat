import logging

import openai
from openai.types.chat import ChatCompletion

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class VectorStoreResult:
    def __init__(self, result_id: str, metrics: dict, content: str):
        self.result_id: str = result_id
        self.metrics: dict = metrics
        self.content: str = content

    def __repr__(self) -> str:
        # return f'[{self.__class__!s}:{self.__dict__!r}]'
        return f'[{self.__class__.__name__}:{self.result_id=!r},{self.metrics=!r},self.content="{self.content[0:20]}{"..." if len(self.content) > 20 else ""}"]'


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
    #
    # token counts: (completion.usage.prompt_tokens, completion.usage.completion_tokens),
    # chat.Chat.check_for_stop_problems(completion)))

    def __init__(self, prompt: str, response_duration_secs: float,
                 completion: ChatCompletion | None, vector_store_response: VectorStoreResponse | None):
        self.prompt: str = prompt
        self.completion: ChatCompletion | None = completion
        self._stop_problems: dict[int, str] = {}
        self.vector_store_response: VectorStoreResponse | None = vector_store_response
        self.response_duration_secs: float = response_duration_secs
        self._overflowed = False

        # calc stop_problems if there's a completion
        if self.completion is not None:
            for i in range(0, len(completion.choices)):
                choice = completion.choices[i]
                stop_problem = ''
                match choice.finish_reason:
                    case 'length':
                        stop_problem = 'too many tokens in request'
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


class ChatExchanges:
    """ a 0-or-more question prompt and list of ChatResponse objects, one for each question """
    _exchanges: list[ChatExchange] = []
    _next_id: int = 1

    def __init__(self, max_exchanges: int):
        """
        initilizes a list to hold at most max_exchanges ChatExchange objects with a fresh id
        :param max_exchanges: when this number of ChatExchange objects is exceeded, oldest is removed
        """
        super().__init__()
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

    def append(self, ce: ChatExchange):
        self._exchanges.append(ce)
        if len(self._exchanges) > self._max_exchanges:
            log.debug(f'reducing overflow id:{self._id}: {len(self._exchanges)} > {self._max_exchanges}')
            self._exchanges.pop(0)
            ce._overflowed = True


class Chat:
    def __init__(self, client: openai.OpenAI):
        self.client = client

    def chat_run_prompt(self, model_name: str, temp: float, max_tokens: int, n: int,
                        sysmsg: str, prompt: str, convo: ChatExchanges):
        # todo: detect stop problems, e.g. not enough tokens
        messages = [{'role': 'system', 'content': sysmsg}]
        for exchange in convo.list():
            # todo: what about previous vector-store responses?
            if exchange.completion is not None:
                messages.append({'role': 'user', 'content': exchange.prompt})
                for choice in exchange.completion.choices:
                    messages.append({'role': choice.message.role, 'content': choice.message.content})
        messages.append({'role': 'user', 'content': prompt})

        completion: ChatCompletion = self.client.chat.completions.create(
            model=model_name,
            temperature=temp,  # default 1.0, 0.0->2.0
            messages=messages,
            max_tokens=max_tokens,  # default 16?
            n=n,

            stream=False,

            # seed=27,
            # top_p=1,  # default 1, ~0.01->1.0
            # frequency_penalty=1,  # default 0, -2.0->2.0
            # presence_penalty=1,  # default 0, -2.0->2.0
            # stop=[],
        )

        return completion
