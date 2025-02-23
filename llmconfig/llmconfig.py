import logging
from abc import ABC, abstractmethod
from collections import OrderedDict

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


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
        self.system_message = LLMConfig.sysmsg_all[init_system_message_name]

    @abstractmethod
    def numbers_oneline_logging_str(self) -> str:
        pass

    @abstractmethod
    def texts_oneline_logging_str(self) -> str:
        pass


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
        self.provider_name = provider_name

    def provider(self) -> str:
        return self.provider_name

    @abstractmethod
    def copy_settings(self) -> LLMSettings:
        pass
