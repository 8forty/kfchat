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
        """
        returns a string of just the numeric settings and their values (i.e. it's short-ish)
        """
        pass

    @abstractmethod
    def texts_oneline_logging_str(self) -> str:
        """
        returns a string of just the text settings and their values (i.e. it's probably long)
        """
        pass

    @dataclass
    class SettingsSpec:
        label: str
        options: Union[list, dict]
        value: any
        tooltip: str

    @abstractmethod
    def specs(self) -> list[SettingsSpec]:
        pass

    def value(self, label: str) -> any:
        """
        returns the current value of setting with given label, or None if there isn't one
        :param label:
        :return:
        """
        for spec in self.specs():
            if spec.label == label:
                return spec.value
        return None

    @abstractmethod
    async def change(self, label: str, value: any) -> None:
        """
        changes the value of the setting with given label to the given value
        :param label:
        :param value:
        """
        pass
