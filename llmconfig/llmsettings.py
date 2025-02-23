from abc import abstractmethod, ABC


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

    @abstractmethod
    def numbers_oneline_logging_str(self) -> str:
        pass

    @abstractmethod
    def texts_oneline_logging_str(self) -> str:
        pass
