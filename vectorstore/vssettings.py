from abc import abstractmethod, ABC

import config


class VSSettings(ABC):
    def __init__(self, init_n: int):
        """
        standard set of settings for vectorstores
        :param init_n:

        """
        super().__init__()
        self.n = init_n

    def __repr__(self) -> str:
        return f'{self.__class__!s}:{self.__dict__!r}'
        # return f'{self.__class__!s}:{self.result_id=!r},{self.metrics=!r},self.content="{self.content[0:20]}{"..." if len(self.content) > 20 else ""}"'

    def __str__(self) -> str:
        return f'{self.__dict__}'

    def numbers_oneline_logging_str(self) -> str:
        return f'n:{self.n}'

    def texts_oneline_logging_str(self) -> str:
        return f''
