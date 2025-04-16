import datetime
import logging
import random
import time
import timeit
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
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

sql_path = 'c:\\sqlite\\kfchat\\kfchat.sqlite3'
sql_chunks_table_name = 'chunks'
sql_chunks_create = f"""
        create table if not exists {sql_chunks_table_name} (
            collection text,
            content    text,
            id         text,
            metadata   text,
            sqlid      integer primary key, -- becomes the rowid
            unique (collection, id)
        );
    """
sql_chunks_insert_trigger_create = f"create trigger if not exists {sql_chunks_table_name}_ai after insert on {sql_chunks_table_name}"
sql_chunks_delete_trigger_create = f"create trigger if not exists {sql_chunks_table_name}_ad after delete on {sql_chunks_table_name}"
sql_chunks_update_trigger_create = f"create trigger if not exists {sql_chunks_table_name}_au after update on {sql_chunks_table_name}"


@dataclass
class FTSSpec:
    table_name: str
    create: str
    insert_trigger: str
    delete_trigger: str
    update_trigger: str
    search: str


class FTSType(Enum):
    SQLITE3_UNICODE = 'sqlite3_unicode61_defaults'
    SQLITE3_UNICODE_IMPROVED = 'sqlite3_unicode61_improved'
    SQLITE3_PORTER_IMPROVED = 'sqlite3_porter_improved'
    SQLITE3_TRIGRAM_IMPROVED = 'sqlite3_trigram_improved'

    @classmethod
    def members(cls) -> list:
        return list(cls.__members__.values())

    @classmethod
    def names(cls) -> list[str]:
        return list(cls.__members__.keys())


def tn(fts_type: FTSType) -> str:
    return f'{sql_chunks_table_name}_{fts_type.value}_fts5'


# these include all the contentless table stuff, including the necessary triggers
sql_chunks_fts5 = {
    FTSType.SQLITE3_UNICODE: FTSSpec(
        table_name=tn(FTSType.SQLITE3_UNICODE),
        create=
        f"""
            create virtual table if not exists {tn(FTSType.SQLITE3_UNICODE)}
                using fts5 (
                collection unindexed,
                content,
                id unindexed,
                metadata,
                content='{sql_chunks_table_name}', -- external content table
                content_rowid='sqlid',
                tokenize = "unicode61 remove_diacritics 2", -- we use remove_diacritics 2 as the default b/c greenfield
            );
        """,
        insert_trigger=
        f"""
                insert into {tn(FTSType.SQLITE3_UNICODE)}(collection, content, id, metadata) values (new.collection, new.content, new.id, new.metadata);
        """,
        delete_trigger=
        f"""
                -- this is the fancy delete command for contentless external content tables: https://www.sqlite.org/fts5.html#the_delete_command 
                insert into {tn(FTSType.SQLITE3_UNICODE)}({tn(FTSType.SQLITE3_UNICODE)}, rowid, collection, content, id, metadata) values ('delete', old.sqlid, old.collection, old.content, old.id, old.metadata);
        """,
        update_trigger=
        f"""
                insert into {tn(FTSType.SQLITE3_UNICODE)}({tn(FTSType.SQLITE3_UNICODE)}, rowid, collection, content, id, metadata) values ('delete', old.sqlid, old.collection, old.content, old.id, old.metadata);
                insert into {tn(FTSType.SQLITE3_UNICODE)}(collection, content, id, metadata) values (new.collection, new.content, new.id, new.metadata);
        """,
        search=f"select substr(content, 1, 40), bm25({tn(FTSType.SQLITE3_UNICODE)}, 0, 1, 0, 0) bm25 from {tn(FTSType.SQLITE3_UNICODE)} where content match '%s';"
    ),

    FTSType.SQLITE3_UNICODE_IMPROVED: FTSSpec(
        table_name=tn(FTSType.SQLITE3_UNICODE_IMPROVED),
        create=
        f"""
            create virtual table if not exists {tn(FTSType.SQLITE3_UNICODE_IMPROVED)}
                using fts5 (
                collection unindexed,
                content,
                id unindexed,
                metadata,
                content='{sql_chunks_table_name}', -- external content table
                content_rowid='sqlid',
                tokenize = "unicode61 remove_diacritics 2 tokenchars '-_'", -- we use remove_diacritics 2 as the default b/c greenfield
            );
        """,
        insert_trigger=
        f"""
                insert into {tn(FTSType.SQLITE3_UNICODE_IMPROVED)}(collection, content, id, metadata) values (new.collection, new.content, new.id, new.metadata);
        """,
        delete_trigger=
        f"""
                -- this is the fancy delete command for contentless external content tables: https://www.sqlite.org/fts5.html#the_delete_command 
                insert into {tn(FTSType.SQLITE3_UNICODE_IMPROVED)}({tn(FTSType.SQLITE3_UNICODE_IMPROVED)}, rowid, collection, content, id, metadata) values ('delete', old.sqlid, old.collection, old.content, old.id, old.metadata);
        """,
        update_trigger=
        f"""
                insert into {tn(FTSType.SQLITE3_UNICODE_IMPROVED)}({tn(FTSType.SQLITE3_UNICODE_IMPROVED)}, rowid, collection, content, id, metadata) values ('delete', old.sqlid, old.collection, old.content, old.id, old.metadata);
                insert into {tn(FTSType.SQLITE3_UNICODE_IMPROVED)}(collection, content, id, metadata) values (new.collection, new.content, new.id, new.metadata);
        """,
        search=f"select substr(content, 1, 40), bm25({tn(FTSType.SQLITE3_UNICODE_IMPROVED)}, 0, 1, 0, 0) bm25 from {tn(FTSType.SQLITE3_UNICODE_IMPROVED)} where content match '%s';"
    ),

    FTSType.SQLITE3_PORTER_IMPROVED: FTSSpec(
        table_name=tn(FTSType.SQLITE3_PORTER_IMPROVED),
        create=
        f"""
            create virtual table if not exists {tn(FTSType.SQLITE3_PORTER_IMPROVED)}
                using fts5 (
                collection unindexed,
                content,
                id unindexed,
                metadata,
                content='{sql_chunks_table_name}', -- external content table
                content_rowid='sqlid',
                tokenize = "porter unicode61 remove_diacritics 2 tokenchars '-_'", -- we use remove_diacritics 2 as the default b/c greenfield
            );
        """,
        insert_trigger=
        f"""
                insert into {tn(FTSType.SQLITE3_PORTER_IMPROVED)}(collection, content, id, metadata) values (new.collection, new.content, new.id, new.metadata);
        """,
        delete_trigger=
        f"""
                -- this is the fancy delete command for contentless external content tables: https://www.sqlite.org/fts5.html#the_delete_command 
                insert into {tn(FTSType.SQLITE3_PORTER_IMPROVED)}({tn(FTSType.SQLITE3_PORTER_IMPROVED)}, rowid, collection, content, id, metadata) values ('delete', old.sqlid, old.collection, old.content, old.id, old.metadata);
        """,
        update_trigger=
        f"""
                insert into {tn(FTSType.SQLITE3_PORTER_IMPROVED)}({tn(FTSType.SQLITE3_PORTER_IMPROVED)}, rowid, collection, content, id, metadata) values ('delete', old.sqlid, old.collection, old.content, old.id, old.metadata);
                insert into {tn(FTSType.SQLITE3_PORTER_IMPROVED)}(collection, content, id, metadata) values (new.collection, new.content, new.id, new.metadata);
        """,
        search=f"select substr(content, 1, 40), bm25({tn(FTSType.SQLITE3_PORTER_IMPROVED)}, 0, 1, 0, 0) bm25 from {tn(FTSType.SQLITE3_PORTER_IMPROVED)} where content match '%s';"
    ),

    FTSType.SQLITE3_TRIGRAM_IMPROVED: FTSSpec(
        table_name=tn(FTSType.SQLITE3_TRIGRAM_IMPROVED),
        create=
        f"""
        create virtual table if not exists {tn(FTSType.SQLITE3_TRIGRAM_IMPROVED)}
            using fts5 (
            collection unindexed,
            content,
            id unindexed,
            metadata,
            content='{sql_chunks_table_name}', -- external content table
            content_rowid='sqlid',
            tokenize = "trigram remove_diacritics 1 case_sensitive 0", -- we use remove_diacritics 2 as the default b/c greenfield
        );
        """,
        insert_trigger=
        f"""
            insert into {tn(FTSType.SQLITE3_TRIGRAM_IMPROVED)}(collection, content, id, metadata) values (new.collection, new.content, new.id, new.metadata);
        """,
        delete_trigger=
        f"""
            -- this is the fancy delete command for contentless external content tables: https://www.sqlite.org/fts5.html#the_delete_command 
            insert into {tn(FTSType.SQLITE3_TRIGRAM_IMPROVED)}({tn(FTSType.SQLITE3_TRIGRAM_IMPROVED)}, rowid, collection, content, id, metadata) values ('delete', old.sqlid, old.collection, old.content, old.id, old.metadata);
        """,
        update_trigger=
        f"""
            insert into {tn(FTSType.SQLITE3_TRIGRAM_IMPROVED)}({tn(FTSType.SQLITE3_TRIGRAM_IMPROVED)}, rowid, collection, content, id, metadata) values ('delete', old.sqlid, old.collection, old.content, old.id, old.metadata);
            insert into {tn(FTSType.SQLITE3_TRIGRAM_IMPROVED)}(collection, content, id, metadata) values (new.collection, new.content, new.id, new.metadata);
        """,
        search=f"select substr(content, 1, 40), bm25({tn(FTSType.SQLITE3_TRIGRAM_IMPROVED)}, 0, 1, 0, 0) bm25 from {tn(FTSType.SQLITE3_TRIGRAM_IMPROVED)} where content match '%s';"
    ),
}


@dataclass
class ModelSpec:
    name: str
    provider: str
    api: Literal['openai', 'anthropic']
    supported_parms: list = field(default_factory=list)  # ['temperature', 'top_p', 'max_tokens', 'n', 'seed', 'system']
    # todo: handle supported_parms, and do so generically


class LLMData:
    models = [

        # todo: config-file these
        ModelSpec('claude-3-5-haiku-20241022', provider='ANTHROPIC', api='anthropic'),
        ModelSpec('claude-3-5-sonnet-20241022', provider='ANTHROPIC', api='anthropic'),

        ModelSpec('RFI-Automate-GPT-4o-mini-2000k', provider='AZURE', api='openai'),

        ModelSpec('gemini-1.5-flash', provider='GEMINI', api='openai'),
        ModelSpec('gemini-1.5-flash-8b', provider='GEMINI', api='openai'),
        ModelSpec('gemini-1.5-pro', provider='GEMINI', api='openai'),
        ModelSpec('gemini-2.0-flash', provider='GEMINI', api='openai'),
        ModelSpec('gemini-2.0-flash-lite', provider='GEMINI', api='openai'),
        ModelSpec('gemini-2.0-pro', provider='GEMINI', api='openai'),
        ModelSpec('gemini-2.0-flash-thinking-exp-01-21', provider='GEMINI', api='openai'),
        ModelSpec('gemini-2.5-pro-exp-03-25', provider='GEMINI', api='openai'),
        ModelSpec('gemma-3-27b-it', provider='GEMINI', api='openai'),

        ModelSpec('gpt-4o-mini', provider='GITHUB', api='openai'),
        ModelSpec('gpt-4o', provider='GITHUB', api='openai'),
        ModelSpec('deepseek-r1', provider='GITHUB', api='openai'),
        ModelSpec('Phi-4', provider='GITHUB', api='openai'),
        ModelSpec('AI21-Jamba-1.5-Large', provider='GITHUB', api='openai'),
        ModelSpec('Cohere-command-r-08-2024', provider='GITHUB', api='openai'),
        ModelSpec('Cohere-command-r-plus-08-2024', provider='GITHUB', api='openai'),
        ModelSpec('Llama-3.3-70B-Instruct', provider='GITHUB', api='openai'),
        ModelSpec('Mistral-Large-2411', provider='GITHUB', api='openai'),

        ModelSpec('grok-2-1212', provider='XAI', api='openai'),

        ModelSpec('llama-3.3-70b-versatile', provider='GROQ', api='openai'),
        ModelSpec('deepseek-r1-distill-llama-70b', provider='GROQ', api='openai'),
        ModelSpec('gemma2-9b-it', provider='GROQ', api='openai'),
        ModelSpec('qwen-qwq-32b', provider='GROQ', api='openai'),
        ModelSpec('meta-llama/llama-4-maverick-17b-128e-instruct', provider='GROQ', api='openai'),
        ModelSpec('meta-llama/llama-4-scout-17b-16e-instruct', provider='GROQ', api='openai'),

        ModelSpec('llama-3.3-70b', provider='CEREBRAS', api='openai'),
        ModelSpec('llama-4-scout-17b-16e-instruct', provider='CEREBRAS', api='openai'),

        ModelSpec('llama3.2:1b', provider='OLLAMA', api='openai'),
        ModelSpec('llama3.2:3b', provider='OLLAMA', api='openai'),
        ModelSpec('mistral-nemo:12b', provider='OLLAMA', api='openai'),
        ModelSpec('mixtral:8x7b', provider='OLLAMA', api='openai'),
        ModelSpec('gemma3:1b', provider='OLLAMA', api='openai'),
        ModelSpec('gemma3:4b', provider='OLLAMA', api='openai'),
        ModelSpec('gemma3:12b', provider='OLLAMA', api='openai'),
        ModelSpec('gemma3:27b', provider='OLLAMA', api='openai'),
        ModelSpec('llama3.3:70b', provider='OLLAMA', api='openai'),
        ModelSpec('llama3.3:70b-instruct-q2_K', provider='OLLAMA', api='openai'),
        ModelSpec('deepseek-r1:1.5b', provider='OLLAMA', api='openai'),
        ModelSpec('deepseek-r1:8b', provider='OLLAMA', api='openai'),
        ModelSpec('deepseek-r1:14b', provider='OLLAMA', api='openai'),
        ModelSpec('deepseek-r1:32b', provider='OLLAMA', api='openai'),
        ModelSpec('deepseek-v2:16b', provider='OLLAMA', api='openai'),
        ModelSpec('qwq:latest', provider='OLLAMA', api='openai'),
        ModelSpec('phi4:14b', provider='OLLAMA', api='openai'),
        ModelSpec('granite3.2:2b', provider='OLLAMA', api='openai'),
        ModelSpec('granite3.2:8b', provider='OLLAMA', api='openai'),
        ModelSpec('phi4-mini', provider='OLLAMA', api='openai'),
        ModelSpec('olmo2:7b', provider='OLLAMA', api='openai'),
        ModelSpec('olmo2:13b', provider='OLLAMA', api='openai'),
        ModelSpec('command-r7b', provider='OLLAMA', api='openai'),
        ModelSpec('openthinker:7b', provider='OLLAMA', api='openai'),
        ModelSpec('openthinker:32b', provider='OLLAMA', api='openai'),
        ModelSpec('qwen2.5:0.5b', provider='OLLAMA', api='openai'),
        ModelSpec('qwen2.5:1.5b', provider='OLLAMA', api='openai'),
        ModelSpec('qwen2.5:3b', provider='OLLAMA', api='openai'),
        ModelSpec('qwen2.5:7b', provider='OLLAMA', api='openai'),
        ModelSpec('qwen2.5:14b', provider='OLLAMA', api='openai'),
        ModelSpec('qwen2.5:32b', provider='OLLAMA', api='openai'),
        ModelSpec('qwen2.5:72b', provider='OLLAMA', api='openai'),

        ModelSpec('gpt-4o-mini', provider='OPENAI', api='openai'),
        ModelSpec('gpt-4o', provider='OPENAI', api='openai'),
        ModelSpec('o1', provider='OPENAI', api='openai'),
        ModelSpec('o3-mini', provider='OPENAI', api='openai'),
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
