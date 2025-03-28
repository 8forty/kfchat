import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass

import logstuff
from chatexchanges import VectorStoreResponse
from config import FTSType
from vectorstore.vssettings import VSSettings

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class VSAPI(ABC):
    @dataclass
    class SearchResponse:
        """
        text, score, and raw results lists, matching collections = matching results
        """
        results_text: list[str]
        results_score: list[float]
        results_raw: list[dict]

    def __init__(self, vs_type_name: str, vssettings: VSSettings, parms: dict[str, str]):
        """

        :param vs_type_name: currently: ['azure', 'chroma']
        :param vssettings: VSSettings instance
        :param parms: (possibly env vars) that set needed parms for the api, e.g. key, endpoint...
        """
        if vs_type_name in ['azure', 'chroma']:
            self._vs_type_name = vs_type_name
            # no vssettings variable b/c it will be subtyped in every concrete class
            self.parms = parms
        else:
            raise ValueError(f'{__class__.__name__}: invalid vs type! {vs_type_name}')

    @abstractmethod
    def settings(self) -> VSSettings:
        pass

    @staticmethod
    @abstractmethod
    def create(vs_type_name: str, vssettings: VSSettings, parms: dict[str, str]):
        pass

    def __repr__(self) -> str:
        return f'{self.__class__!s}:{self.__dict__!r}'

    @abstractmethod
    def warmup(self):
        pass

    def type(self) -> str:
        return self._vs_type_name

    @abstractmethod
    def list_collection_names(self) -> list[str]:
        pass

    @abstractmethod
    def raw_search(self, prompt: str, howmany: int) -> SearchResponse:
        pass

    @abstractmethod
    def search(self, prompt: str) -> VectorStoreResponse:
        pass

    @abstractmethod
    def switch_collection(self, new_collection_name: str) -> None:
        pass
