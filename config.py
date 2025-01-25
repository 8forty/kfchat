import datetime
import logging
import random
import time
import timeit

import dotenv

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

random.seed(27)

dotenv.load_dotenv(override=True)
env = dotenv.dotenv_values()

name = 'kfchat'

llm_api_types_config = {
    'azure': {
        'key': env.get('kfAZURE_OPENAI_API_KEY'),
        'AZURE_OPENAI_API_VERSION': env.get('AZURE_OPENAI_API_VERSION'),
        'AZURE_OPENAI_ENDPOINT': env.get('AZURE_OPENAI_ENDPOINT'),
        'AZURE_AI_SEARCH_ENDPOINT': env.get('AZURE_AI_SEARCH_ENDPOINT'),
        'ai-search-api-key': env.get('AZURE_AI_SEARCH_API_KEY'),
    },
    'ollama': {
        'key': 'nokeyneeded',
        'OLLAMA_ENDPOINT': 'http://localhost:11434/v1/',
    },
    'openai': {
        'key': env.get('kfOPENAI_API_KEY'),
        'OPENAI_CHAT_COMPLETIONS_ENDPOINT': env.get('OPENAI_CHAT_COMPLETIONS_ENDPOINT'),
        'OPENAI_ENDPOINT': env.get('OPENAI_ENDPOINT'),
    },
    'groq': {
        'key': env.get('kfGROQ_API_KEY'),
        'GROQ_OPENAI_ENDPOINT': env.get('GROQ_OPENAI_ENDPOINT'),
    },
    'gemini': {
        'key': env.get('kfGEMINI_API_KEY'),
        'GEMINI_OPENAI_ENDPOINT': env.get('GEMINI_OPENAI_ENDPOINT'),
    },
}

chat_exchanges_circular_list_count = 10


def now_datetime() -> str:
    return datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d-%H:%M:%S')


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
