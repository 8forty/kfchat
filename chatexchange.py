import logging
from dataclasses import dataclass
from typing import Tuple, List

import config
import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


@dataclass
class ChatExchange:
    prompt: str
    response: str  # todo: this is only choice[0]
    token_counts: tuple[int, int]  # prompt-tokens, completion-tokens
    duration_seconds: float
    stop_problems: dict[int, str]  # choice # -> stop problem description


class ChatExchanges(list):
    """ a 0-or-more question prompt and list of ChatResponse objects, one for each question """
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

    def id(self):
        return self._id

    def empty(self) -> bool:
        return len(self) == 0

    def add(self, prompt: str, response: str, token_counts: tuple[int, int], duration_seconds: float, stop_problems: dict[int, str]) -> None:
        self.append(ChatExchange(prompt, response, token_counts, duration_seconds, stop_problems))
        if len(self) > self._max_exchanges:
            log.debug(f'reducing overflow id:{self._id}: {len(self)} > {self._max_exchanges}')
            self.pop(0)
