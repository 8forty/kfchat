from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Union


class BaseSettings(ABC):
    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__!s}:{self.__dict__!r}'
        # return f'{self.__class__!s}:{self.result_id=!r},{self.metrics=!r},self.content="{self.content[0:20]}{"..." if len(self.content) > 20 else ""}"'

    def __str__(self) -> str:
        return f'{self.__dict__}'

    @abstractmethod
    def numbers_oneline_logging_str(self) -> str:
        pass

    @abstractmethod
    def texts_oneline_logging_str(self) -> str:
        pass

    @dataclass
    class Info:
        label: str
        options: Union[list, dict]
        value: any
        tooltip: str

    @abstractmethod
    def info(self) -> list[Info]:
        pass

    @abstractmethod
    async def change(self, label: str, value: any) -> None:
        pass
