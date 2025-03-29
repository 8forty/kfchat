import logging

import dotenv
from openai.types.chat import ChatCompletion

import logstuff
from llmconfig.llm_openai_config import LLMOpenAIConfig, LLMOpenAISettings

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

dotenv.load_dotenv(override=True)
env = dotenv.dotenv_values()


# Google Generative AI
class LLMOpenAIGGAIConfig(LLMOpenAIConfig):
    def __init__(self, model_name: str, provider: str, settings: LLMOpenAISettings):
        """

        :param model_name:
        :param provider
        :param settings:

        """
        super().__init__(model_name, provider, settings)

    def generate_chat_completion(self, messages: list[dict]) -> ChatCompletion:
        """
        google generative AI:
            "seed" parameter never allowed
            sysmsg sometimes allowed
        :param messages:
        :return:
        """

        chat_messages = messages
        chat_n = self._settings.n

        # models that don't support 'system' role in messages
        # todo: add restrictions to chat details
        if self.model_name == 'gemma-3-27b-it':
            chat_messages = [m for m in messages if m['role'] != 'system']
            chat_n = 1

        chat_completion: ChatCompletion = self._client().chat.completions.create(
            model=self.model_name,
            temperature=self._settings.temp,  # default 1.0, 0.0->2.0
            top_p=self._settings.top_p,  # default 1, ~0.01->1.0
            messages=chat_messages,
            max_tokens=self._settings.max_tokens,  # default 16?
            n=chat_n,

            stream=False,

        )

        return chat_completion
