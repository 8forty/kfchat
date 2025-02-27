import datetime
import logging
import random
import time
import timeit
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import dotenv

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

random.seed(27)

dotenv.load_dotenv(override=True)
env = dotenv.dotenv_values()

name = 'kfchat'

chat_exchanges_circular_list_count = 10


class LLMData:
    # models
    @dataclass
    class ModelSpec:
        name: str
        provider: str
        api: Literal['openai', 'anthropic']

    models = [
        ModelSpec('gpt-4o-mini', provider='github', api='openai'),
        ModelSpec('gpt-4o', provider='github', api='openai'),
        ModelSpec('deepseek-r1', provider='github', api='openai'),
        ModelSpec('Phi-4', provider='github', api='openai'),
        ModelSpec('AI21-Jamba-1.5-Large', provider='github', api='openai'),
        ModelSpec('Cohere-command-r-08-2024', provider='github', api='openai'),
        ModelSpec('Cohere-command-r-plus-08-2024', provider='github', api='openai'),
        ModelSpec('Llama-3.3-70B-Instruct', provider='github', api='openai'),
        ModelSpec('Mistral-Large-2411', provider='github', api='openai'),

        ModelSpec('llama-3.3-70b-versatile', provider='groq', api='openai'),
        ModelSpec('deepseek-r1-distill-llama-70b', provider='groq', api='openai'),
        ModelSpec('gemma2-9b-it', provider='groq', api='openai'),
        ModelSpec('mixtral-8x7b-32768', provider='groq', api='openai'),

        ModelSpec('llama3.2:1b', provider='ollama', api='openai'),
        ModelSpec('llama3.2:3b', provider='ollama', api='openai'),
        ModelSpec('mistral-nemo:12b', provider='ollama', api='openai'),
        ModelSpec('gemma2:2b', provider='ollama', api='openai'),
        ModelSpec('gemma2:9b', provider='ollama', api='openai'),
        ModelSpec('gemma2:27b', provider='ollama', api='openai'),
        ModelSpec('llama3.3:70b', provider='ollama', api='openai'),
        ModelSpec('llama3.3:70b-instruct-q2_K', provider='ollama', api='openai'),
        ModelSpec('deepseek-r1:1.5b', provider='ollama', api='openai'),
        ModelSpec('deepseek-r1:8b', provider='ollama', api='openai'),
        ModelSpec('deepseek-r1:14b', provider='ollama', api='openai'),
        ModelSpec('deepseek-v2:16b', provider='ollama', api='openai'),
        ModelSpec('deepseek-v2:32b', provider='ollama', api='openai'),
        ModelSpec('qwq:latest', provider='ollama', api='openai'),
        ModelSpec('phi4:14b', provider='ollama', api='openai'),

        ModelSpec('gemini-1.5-flash', provider='gemini', api='openai'),
        ModelSpec('gemini-1.5-flash-8b', provider='gemini', api='openai'),
        ModelSpec('gemini-1.5-pro', provider='gemini', api='openai'),
        ModelSpec('gemini-2.0-flash-001', provider='gemini', api='openai'),
        ModelSpec('gemini-2.0-flash-lite-preview-02-05', provider='gemini', api='openai'),
        ModelSpec('gemini-2.0-pro-exp-02-05', provider='gemini', api='openai'),
        ModelSpec('gemini-2.0-flash-thinking-exp-01-21', provider='gemini', api='openai'),

        ModelSpec('gpt-4o-mini', provider='openai', api='openai'),
        ModelSpec('gpt-4o', provider='openai', api='openai'),

        ModelSpec('RFI-Automate-GPT-4o-mini-2000k', provider='azure', api='openai'),

        ModelSpec('claude-3-5-haiku-20241022', provider='anthropic', api='anthropic'),
        ModelSpec('claude-3-5-sonnet-20241022', provider='anthropic', api='anthropic'),
    ]
    models_by_pname = {f'{ms.provider}.{ms.name}': ms for ms in models}
    providers = {ms.provider for ms in models}
    apis = {ms.api for ms in models}
    models_by_provider = {provider: [] for provider in providers}
    models_by_api = {api: [] for api in apis}
    for ms in models:
        models_by_provider[ms.provider].append(ms)
        models_by_api[ms.api].append(ms)

    # system messages
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
    rag1_sysmsg = ('You are a chatbot that answers questions that are labeled "Question:" based on the context labeled "Context:". '
                   'Keep your answers short and concise. '
                   'Always respond "I don\'t know" if you are not sure about the answer.')
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
        'rag1': rag1_sysmsg,
        'text-sentiment': textclass_sysmsg,
        'carl-sagan': csagan_sysmsg,
        'empty': empty_sysmsg,
    })


def now_datetime() -> str:
    return datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d-%H:%M:%S')


def ancient_datetime() -> str:
    return datetime.datetime.fromordinal(1).strftime('%Y-%m-%d-%H:%M:%S')


def now_time() -> str:
    return datetime.datetime.now(datetime.UTC).strftime('%H:%M:%S')


def secs_string(start: float, end: float = None) -> str:
    if end is None:
        end = timeit.default_timer()
    return time.strftime('%H:%M:%S', time.gmtime(end - start))


def redact(secret: str) -> str:
    return f'{secret[0:3]}...[REDACTED]...{secret[-3:]}'


def redact_parms(parms: dict[str, str]) -> dict[str, str]:
    retval = {}
    for k, v in parms.items():
        if 'key' in k.lower():
            retval[k] = redact(v)
        else:
            retval[k] = v

    return retval
