import openai

from llmopenaiapi import LLMOpenaiAPI
from modelapi import ModelAPI


class LLMConfig:
    def __init__(self, model_api: ModelAPI, env_values: dict[str, str], model_name: str, default_temp: float, default_max_tokens: int, default_system_message: str):
        """

        :param model_api:
        :param env_values: parms for the api, e.g. key, endpoint, token...
        :param model_name:
        :param default_temp:
        :param default_max_tokens:
        :param default_system_message:

        """
        # todo: these should come from e.g. pref screen
        self.model_api: ModelAPI = model_api
        self.model_name: str = model_name
        self.default_temp: float = default_temp
        self.default_max_tokens: int = default_max_tokens
        self.default_system_message: str = default_system_message
        self.client: openai.OpenAI = self.model_api.client()
        self.api = LLMOpenaiAPI(self.client)
