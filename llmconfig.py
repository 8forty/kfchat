import logging
from abc import ABC

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMConfig(ABC):
    def __init__(self, model_name: str, api_type_name: str, parms: dict[str, str]):
        """

        :param model_name
        :param api_type_name
        :param parms
        """
        self.model_name = model_name
        self.api_type_name = api_type_name
        self.parms = parms

    def api_type(self) -> str:
        return self.api_type_name
