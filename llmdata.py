import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal, Any

import dotenv
import yaml

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

dotenv.load_dotenv(override=True)
env = dotenv.dotenv_values()


@dataclass
class ModelSpec:
    name: str
    provider: str
    api: Literal['openai', 'anthropic', 'ollama']
    supported_parms: list = field(default_factory=list)  # ['temperature', 'top_p', 'max_tokens', 'n', 'seed', 'system']
    # todo: handle supported_parms, and do so generically


class LLMData:

    def __init__(self):
        # load the models config
        self.models = []
        filename = 'models.yml'  # todo: configure this
        log.debug(f'loading models from {filename}')
        with open(filename, 'r') as efile:
            models_dict: dict[str, dict[str, dict[str, Any]]] = yaml.safe_load(efile)
            models_dict.pop('metadata')  # ignore

            for provider in models_dict:
                log.debug(f'added provider {provider}')
                for model in models_dict[provider]:
                    api = models_dict[provider][model]['api']
                    self.models.append(ModelSpec(name=model, provider=provider, api=api))

        self.models_by_pname = {f'{ms.provider}.{ms.name}': ms for ms in self.models}
        self.providers = {ms.provider for ms in self.models}
        self.apis = {ms.api for ms in self.models}
        self.models_by_provider = {provider: [] for provider in self.providers}
        self.models_by_api = {api: [] for api in self.apis}
        for ms in self.models:
            self.models_by_provider[ms.provider].append(ms)
            self.models_by_api[ms.api].append(ms)

        # system messages
        self.conversational_sysmsg = 'You are a helpful chatbot that talks in a conversational manner.'
        self.conversational80_sysmsg = ('You are a helpful chatbot that talks in a conversational manner. '
                                        'Your responses must always be less than 80 tokens.')
        self.professional_sysmsg = 'You are a helpful chatbot that talks in a professional manner.'
        self.professional80_sysmsg = ('You are a helpful chatbot that talks in a professional manner. '
                                      'Your responses must always be less than 80 tokens.')
        self.professional800_sysmsg = ('You are a helpful chatbot that talks in a professional manner. '
                                       'Your responses must always be less than 800 tokens.')
        self.technical_sysmsg = ('You are an AI research assistant. '
                                 'Respond in a tone that is technical and scientific.')
        self.technical80_sysmsg = ('You are an AI research assistant. '
                                   'Respond in a tone that is technical and scientific.'
                                   'Your responses must always be less than 800 tokens.')
        self.technical800_sysmsg = ('You are an AI research assistant. Respond in a tone that is technical and scientific. '
                                    'All math equations should be latex format delimited by $$. '
                                    'Your responses must always be less than 800 tokens.')
        self.textclass_sysmsg = 'Classify each prompt into neutral, negative or positive.'
        self.csagan_sysmsg = 'You are a helpful assistant that talks like Carl Sagan.'
        self.empty_sysmsg = ''

        self.sysmsg_all = OrderedDict({
            'convo': self.conversational_sysmsg,
            'convo80': self.conversational80_sysmsg,
            'professional': self.professional_sysmsg,
            'professional80': self.professional80_sysmsg,
            'professional800': self.professional800_sysmsg,
            'technical': self.technical_sysmsg,
            'technical80': self.technical80_sysmsg,
            'technical800': self.technical800_sysmsg,
            'text-sentiment': self.textclass_sysmsg,
            'carl-sagan': self.csagan_sysmsg,
            'empty': self.empty_sysmsg,
        })

        # RAG system messages with {context} and {sysmsg}
        self.dont_know = 'I\'m sorry, the given collection of information doesn\'t appear to contain that information.'
        self.rag1_sysmsg = (
                '{sysmsg}'
                'Context information is below.'
                '---------------------'
                '{context}'
                '---------------------'
                'Please answer using the given context only, do not use any prior knowledge.'
                'Always respond "' + self.dont_know +
                '" if you are not sure about the answer.'
                'In any case, don\'t answer using your own knowledge.'
        )

        #     You are a bot that answers questions.  Please answer using retrieved documents only
        #     and without using your knowledge. Please generate citations to retrieved documents for every claim in your
        #     answer.  In any case, don't answer using your own knowledge.  If you don't know an answer, please say "{g_dont_know_preferred}"
