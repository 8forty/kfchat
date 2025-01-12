import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class VSAPI(ABC):
    @dataclass
    class SearchResponse:
        """
        text, score, and raw results lists, matching indexes = matching results
        """
        results_text: list[str]
        results_score: list[float]
        results_raw: list[dict]

    def __init__(self, api_type_name: str, parms: dict[str, str]):
        """

        :param api_type_name: currently: ['azure', 'chroma']
        :param parms: (possibly env vars) that set needed parms for the api, e.g. key, endpoint...
        """
        if api_type_name in ['azure', 'chroma']:
            self._api_type_name = api_type_name
            self.parms = parms
        else:
            raise ValueError(f'{__class__.__name__}: invalid api_type! {api_type_name}')

    @staticmethod
    @abstractmethod
    def create(api_type_name: str, parms: dict[str, str]):
        pass

    def __repr__(self) -> str:
        return f'[{self.__class__!s}:{self.__dict__!r}]'

    @abstractmethod
    def warmup(self):
        pass

    def type(self) -> str:
        return self._api_type_name

    @abstractmethod
    def list_index_names(self) -> list[str]:
        pass

    @abstractmethod
    def search(self, prompt: str, howmany: int) -> SearchResponse:
        pass

    @abstractmethod
    def change_index(self, new_index_name: str) -> None:
        pass
