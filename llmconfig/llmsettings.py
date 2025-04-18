from abc import ABC

from basesettings import BaseSettings


class LLMSettings(BaseSettings, ABC):

    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__!s}:{self.__dict__!r}'
        # return f'{self.__class__!s}:{self.result_id=!r},{self.metrics=!r},self.content="{self.content[0:20]}{"..." if len(self.content) > 20 else ""}"'

    def __str__(self) -> str:
        return f'{self.__dict__}'
