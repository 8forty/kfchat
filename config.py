import datetime
import logging
import random
import time
import timeit
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Any

import dotenv
import yaml

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

random.seed(27)

dotenv.load_dotenv(override=True)
env = dotenv.dotenv_values()

name = 'kfchat'

chat_exchanges_circular_list_count = 10

sql_path = 'c:/sqlite/kfchat/kfchat.sqlite3'  # slashes work ok on windows
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
# noinspection SqlIdentifier
sql_chunks_insert_trigger_create = f"create trigger if not exists {sql_chunks_table_name}_ai after insert on {sql_chunks_table_name}"
# noinspection SqlIdentifier
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
        create=f"""
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
        insert_trigger=f"""
                insert into {tn(FTSType.SQLITE3_UNICODE)}(collection, content, id, metadata) values (new.collection, new.content, new.id, new.metadata);
        """,
        delete_trigger=f"""
                -- this is the fancy delete command for contentless external content tables: https://www.sqlite.org/fts5.html#the_delete_command 
                insert into {tn(FTSType.SQLITE3_UNICODE)}({tn(FTSType.SQLITE3_UNICODE)}, rowid, collection, content, id, metadata) values ('delete', old.sqlid, old.collection, old.content, old.id, old.metadata);
        """,
        update_trigger=f"""
                insert into {tn(FTSType.SQLITE3_UNICODE)}({tn(FTSType.SQLITE3_UNICODE)}, rowid, collection, content, id, metadata) values ('delete', old.sqlid, old.collection, old.content, old.id, old.metadata);
                insert into {tn(FTSType.SQLITE3_UNICODE)}(collection, content, id, metadata) values (new.collection, new.content, new.id, new.metadata);
        """,
        search=f"select substr(content, 1, 40), bm25({tn(FTSType.SQLITE3_UNICODE)}, 0, 1, 0, 0) bm25 from {tn(FTSType.SQLITE3_UNICODE)} where content match '%s';"
    ),

    FTSType.SQLITE3_UNICODE_IMPROVED: FTSSpec(
        table_name=tn(FTSType.SQLITE3_UNICODE_IMPROVED),
        create=f"""
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
        insert_trigger=f"""
                insert into {tn(FTSType.SQLITE3_UNICODE_IMPROVED)}(collection, content, id, metadata) values (new.collection, new.content, new.id, new.metadata);
        """,
        delete_trigger=f"""
                -- this is the fancy delete command for contentless external content tables: https://www.sqlite.org/fts5.html#the_delete_command 
                insert into {tn(FTSType.SQLITE3_UNICODE_IMPROVED)}({tn(FTSType.SQLITE3_UNICODE_IMPROVED)}, rowid, collection, content, id, metadata) values ('delete', old.sqlid, old.collection, old.content, old.id, old.metadata);
        """,
        update_trigger=f"""
                insert into {tn(FTSType.SQLITE3_UNICODE_IMPROVED)}({tn(FTSType.SQLITE3_UNICODE_IMPROVED)}, rowid, collection, content, id, metadata) values ('delete', old.sqlid, old.collection, old.content, old.id, old.metadata);
                insert into {tn(FTSType.SQLITE3_UNICODE_IMPROVED)}(collection, content, id, metadata) values (new.collection, new.content, new.id, new.metadata);
        """,
        search=f"select substr(content, 1, 40), bm25({tn(FTSType.SQLITE3_UNICODE_IMPROVED)}, 0, 1, 0, 0) bm25 from {tn(FTSType.SQLITE3_UNICODE_IMPROVED)} where content match '%s';"
    ),

    FTSType.SQLITE3_PORTER_IMPROVED: FTSSpec(
        table_name=tn(FTSType.SQLITE3_PORTER_IMPROVED),
        create=f"""
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
        insert_trigger=f"""
                insert into {tn(FTSType.SQLITE3_PORTER_IMPROVED)}(collection, content, id, metadata) values (new.collection, new.content, new.id, new.metadata);
        """,
        delete_trigger=f"""
                -- this is the fancy delete command for contentless external content tables: https://www.sqlite.org/fts5.html#the_delete_command 
                insert into {tn(FTSType.SQLITE3_PORTER_IMPROVED)}({tn(FTSType.SQLITE3_PORTER_IMPROVED)}, rowid, collection, content, id, metadata) values ('delete', old.sqlid, old.collection, old.content, old.id, old.metadata);
        """,
        update_trigger=f"""
                insert into {tn(FTSType.SQLITE3_PORTER_IMPROVED)}({tn(FTSType.SQLITE3_PORTER_IMPROVED)}, rowid, collection, content, id, metadata) values ('delete', old.sqlid, old.collection, old.content, old.id, old.metadata);
                insert into {tn(FTSType.SQLITE3_PORTER_IMPROVED)}(collection, content, id, metadata) values (new.collection, new.content, new.id, new.metadata);
        """,
        search=f"select substr(content, 1, 40), bm25({tn(FTSType.SQLITE3_PORTER_IMPROVED)}, 0, 1, 0, 0) bm25 from {tn(FTSType.SQLITE3_PORTER_IMPROVED)} where content match '%s';"
    ),

    FTSType.SQLITE3_TRIGRAM_IMPROVED: FTSSpec(
        table_name=tn(FTSType.SQLITE3_TRIGRAM_IMPROVED),
        create=f"""
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
        insert_trigger=f"""
            insert into {tn(FTSType.SQLITE3_TRIGRAM_IMPROVED)}(collection, content, id, metadata) values (new.collection, new.content, new.id, new.metadata);
        """,
        delete_trigger=f"""
            -- this is the fancy delete command for contentless external content tables: https://www.sqlite.org/fts5.html#the_delete_command 
            insert into {tn(FTSType.SQLITE3_TRIGRAM_IMPROVED)}({tn(FTSType.SQLITE3_TRIGRAM_IMPROVED)}, rowid, collection, content, id, metadata) values ('delete', old.sqlid, old.collection, old.content, old.id, old.metadata);
        """,
        update_trigger=f"""
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


llm_data = LLMData()


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
