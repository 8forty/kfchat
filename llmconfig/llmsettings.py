from abc import abstractmethod, ABC

import config


class LLMSettings(ABC):
    def __init__(self, init_n: int, init_temp: float, init_top_p: float, init_max_tokens: int, init_system_message_name: str):
        """
        (almost) standard set of settings for LLMs
        todo: some LLMs/providers don't support n
        :param init_n:
        :param init_temp:
        :param init_top_p:
        :param init_max_tokens:
        :param init_system_message_name:

        """
        super().__init__()
        self.n = init_n
        self.temp = init_temp
        self.top_p = init_top_p
        self.max_tokens = init_max_tokens
        self.system_message_name = init_system_message_name
        self.system_message = config.LLMData.sysmsg_all[init_system_message_name]

    def __repr__(self) -> str:
        return f'{self.__class__!s}:{self.__dict__!r}'
        # return f'{self.__class__!s}:{self.result_id=!r},{self.metrics=!r},self.content="{self.content[0:20]}{"..." if len(self.content) > 20 else ""}"'

    def __str__(self) -> str:
        return f'{self.__dict__}'

    def numbers_oneline_logging_str(self) -> str:
        return f'n:{self.n},temp:{self.temp},top_p:{self.top_p},max_tokens:{self.max_tokens}'

    def texts_oneline_logging_str(self) -> str:
        return f'sysmsg:{self.system_message}'
