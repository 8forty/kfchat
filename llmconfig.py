import logging
from abc import ABC
from collections import OrderedDict

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMConfig(ABC):
    def __init__(self, model_name: str, provider_name: str):
        """

        :param model_name
        :param provider_name
        """
        self.model_name = model_name
        self.provider_name = provider_name

        self.conversational_sysmsg = 'You are a helpful chatbot that talks in a conversational manner.'
        self.conversational80_sysmsg = ('You are a helpful chatbot that talks in a conversational manner. '
                                        'Your responses must always be less than 80 tokens.')
        self.professional_sysmsg = 'You are a helpful chatbot that talks in a conversational manner.'
        self.professional80_sysmsg = ('You are a helpful chatbot that talks in a conversational manner. '
                                      'Your responses must always be less than 80 tokens.')
        self.professional800_sysmsg = ('You are a helpful chatbot that talks in a conversational manner. '
                                       'Your responses must always be less than 800 tokens.')
        self.technical_sysmsg = ('You are an AI research assistant. '
                                 'Respond in a tone that is technical and scientific.')
        self.technical800_sysmsg = ('You are an AI research assistant. Respond in a tone that is technical and scientific. '
                                    'All math equations should be latex format delimited by $$. '
                                    'Your responses must always be less than 800 tokens.')
        self.answer_sysmsg = ('You are a chatbot that answers questions that are labeled "Question:" based on the context labeled "Context:". '
                              'Keep your answers short and concise. '
                              'Always respond "Unsure about answer" if not sure about the answer.')
        self.textclass_sysmsg = 'Classify each prompt into neutral, negative or positive.'
        self.csagan_sysmsg = 'You are a helpful assistant that talks like Carl Sagan.'
        self.empty_sysmsg = ''

        self.sysmsg_all = OrderedDict({
            'generic': self.conversational_sysmsg,
            'generic80': self.conversational80_sysmsg,
            'professional': self.professional_sysmsg,
            'professional80': self.professional80_sysmsg,
            'professional800': self.professional800_sysmsg,
            'technical': self.technical_sysmsg,
            'technical800': self.technical800_sysmsg,
            'answer': self.answer_sysmsg,
            'text-sentiment': self.textclass_sysmsg,
            'carl-sagan': self.csagan_sysmsg,
            'empty': self.empty_sysmsg,
        })

    def provider(self) -> str:
        return self.provider_name
