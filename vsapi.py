import logging
from dataclasses import dataclass

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class VSAPI:
    @dataclass
    class SearchResponse:
        """
        text, score, and raw results lists, matching indexes = matching results
        """
        results_text: list[str]
        results_score: list[float]
        results_raw: list[dict]

    def __init__(self, api_type: str, index_name: str, parms: dict[str, str]):
        """

        :param api_type: currently: ['azure', 'chroma']
        :param index_name
        :param parms: (possibly env vars) that set needed parms for the api, e.g. key, endpoint...
        """
        if api_type in ['azure', 'chroma']:
            self._api_type_name = api_type
            self.index_name = index_name
            self.parms = parms
        else:
            raise ValueError(f'{__class__.__name__}: invalid api_type! {api_type}')

    def __repr__(self) -> str:
        return f'[{self.__class__!s}:{self.__dict__!r}]'

    def type(self) -> str:
        return self._api_type_name

    def search(self, prompt: str, howmany: int) -> SearchResponse:
        pass
