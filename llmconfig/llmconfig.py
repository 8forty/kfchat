import logging
from abc import ABC, abstractmethod
from collections import OrderedDict

import logstuff
from llmconfig.llmexchange import LLMExchange, LLMMessagePair
from llmconfig.llmsettings import LLMSettings

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMConfig(ABC):
    conversational_sysmsg = 'You are a helpful chatbot that talks in a conversational manner.'
    conversational80_sysmsg = ('You are a helpful chatbot that talks in a conversational manner. '
                               'Your responses must always be less than 80 tokens.')
    professional_sysmsg = 'You are a helpful chatbot that talks in a professional manner.'
    professional80_sysmsg = ('You are a helpful chatbot that talks in a professional manner. '
                             'Your responses must always be less than 80 tokens.')
    professional800_sysmsg = ('You are a helpful chatbot that talks in a professional manner. '
                              'Your responses must always be less than 800 tokens.')
    technical_sysmsg = ('You are an AI research assistant. '
                        'Respond in a tone that is technical and scientific.')
    technical800_sysmsg = ('You are an AI research assistant. Respond in a tone that is technical and scientific. '
                           'All math equations should be latex format delimited by $$. '
                           'Your responses must always be less than 800 tokens.')
    answer_sysmsg = ('You are a chatbot that answers questions that are labeled "Question:" based on the context labeled "Context:". '
                     'Keep your answers short and concise. '
                     'Always respond "Unsure about answer" if not sure about the answer.')
    textclass_sysmsg = 'Classify each prompt into neutral, negative or positive.'
    csagan_sysmsg = 'You are a helpful assistant that talks like Carl Sagan.'
    empty_sysmsg = ''

    sysmsg_all = OrderedDict({
        'generic': conversational_sysmsg,
        'generic80': conversational80_sysmsg,
        'professional': professional_sysmsg,
        'professional80': professional80_sysmsg,
        'professional800': professional800_sysmsg,
        'technical': technical_sysmsg,
        'technical800': technical800_sysmsg,
        'answer': answer_sysmsg,
        'text-sentiment': textclass_sysmsg,
        'carl-sagan': csagan_sysmsg,
        'empty': empty_sysmsg,
    })

    def __init__(self, model_name: str, provider_name: str):
        """

        :param model_name
        :param provider_name
        """
        self.model_name = model_name
        self._provider = provider_name

    def provider(self) -> str:
        return self._provider

    @abstractmethod
    async def change_n(self, new_n: int):
        pass

    @abstractmethod
    async def change_temp(self, new_temp: float):
        pass

    @abstractmethod
    async def change_top_p(self, new_top_p: float):
        pass

    @abstractmethod
    async def change_max_tokens(self, new_max_tokens: int):
        pass

    @abstractmethod
    async def change_sysmsg(self, new_system_message_name: str):
        pass

    @abstractmethod
    def settings(self) -> LLMSettings:
        pass

    @abstractmethod
    def copy_settings(self) -> LLMSettings:
        pass

    @staticmethod
    def _clean_messages(messages: list[LLMMessagePair]) -> list[LLMMessagePair]:
        for message in messages:
            if message.role == 'system':
                log.warning(f"'system' message removed from messages! {message}")
        return [mp for mp in messages if mp.role != 'system']

    @abstractmethod
    def chat_messages(self, messages: list[LLMMessagePair]) -> LLMExchange:
        """
        run chat-completion from a list of messages that includes the prompt as a final dict: {role': 'user', 'content': '...'}
        NOTE: this configuration's system message will be used instead of any supplied in messages
        :param messages:
        """

    pass

    @abstractmethod
    def chat_convo(self, convo: list[LLMExchange], prompt: str) -> LLMExchange:
        """
        run chat-completion
        NOTE: this configuration's system message will be used instead of any supplied in convo
        :param convo: properly ordered list of LLMOpenaiExchange's
        :param prompt: the prompt duh
        """
        pass
