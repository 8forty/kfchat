import logging
from abc import ABC

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMConfig(ABC):
    def __init__(self, model_name: str, provider_name: str):
        """

        :param model_name
        :param provider_name
        """
        self.model_name = model_name
        self.provider_name = provider_name

    def provider(self) -> str:
        return self.provider_name
