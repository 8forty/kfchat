import openai

from llmapi import LLMAPI


class LLMConfig:
    def __init__(self, model_api: LLMAPI, model_name: str, default_temp: float, default_max_tokens: int, default_system_message: str):
        """

        :param model_api:
        :param model_name:
        :param default_temp:
        :param default_max_tokens:
        :param default_system_message:

        """
        self.model_api: LLMAPI = model_api
        self.client: openai.OpenAI = self.model_api.client()

        self.model_name: str = model_name
        self.temp: float = default_temp
        self.max_tokens: int = default_max_tokens
        self.system_message: str = default_system_message
