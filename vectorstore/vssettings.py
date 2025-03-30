from abc import abstractmethod, ABC

from basesettings import BaseSettings


class VSSettings(BaseSettings, ABC):
    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__!s}:{self.__dict__!r}'
        # return f'{self.__class__!s}:{self.result_id=!r},{self.metrics=!r},self.content="{self.content[0:20]}{"..." if len(self.content) > 20 else ""}"'

    def __str__(self) -> str:
        return f'{self.__dict__}'

    def numbers_oneline_logging_str(self) -> str:
        return f'n:{self.n}'

    def texts_oneline_logging_str(self) -> str:
        return f''

    def specs(self) -> list[BaseSettings.SettingsSpec]:
        return [
            BaseSettings.SettingsSpec(label='n', options=[i for i in range(1, 10)], value=self.n),
        ]
