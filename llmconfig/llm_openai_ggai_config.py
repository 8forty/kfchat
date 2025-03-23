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
        google generative AI: no "seed" parameter allowed
        :param messages:
        :return:
        """
        chat_completion: ChatCompletion = self._client().chat.completions.create(
            model=self.model_name,
            temperature=self._settings.temp,  # default 1.0, 0.0->2.0
            top_p=self._settings.top_p,  # default 1, ~0.01->1.0
            messages=messages,
            max_tokens=self._settings.max_tokens,  # default 16?
            n=self._settings.n,

            stream=False,

        )
        return chat_completion
