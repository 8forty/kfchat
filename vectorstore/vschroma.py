import json
import logging
import sqlite3
import timeit
from typing import Annotated

import chromadb
import chromadb.api.types as chroma_api_types
from chromadb.api.models.Collection import Collection
from chromadb.errors import InvalidCollectionException
from chromadb.utils.embedding_functions.google_embedding_function import GoogleGenerativeAiEmbeddingFunction
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import SentenceTransformerEmbeddingFunction
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import Field, validate_call

import config
import logstuff
from chatexchanges import VectorStoreResponse, VectorStoreResult
from config import FTSType
from langchain import lc_docloaders, lc_chunkers
from vectorstore.vsapi import VSAPI
from vectorstore.vschroma_settings import VSChromaSettings
from vectorstore.vssettings import VSSettings

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

# function-name-string : {model-name : {function, create_parms: {model_name}, read_parms: {}}}
chroma_embedding_types: dict[str, dict[str, dict[str, any]]] = {
    'SentenceTransformer-Embeddings': {
        'all-MiniLM-L6-v2': {
            'function': SentenceTransformerEmbeddingFunction,
            'create_parms': {'model_name': 'all-MiniLM-L6-v2'},
            'read_parms': {},
        },
        'all-mpnet-base-v2': {
            'function': SentenceTransformerEmbeddingFunction,
            'create_parms': {'model_name': 'all-mpnet-base-v2'},
            'read_parms': {},
        },
    },
    'OpenAI-Embeddings': {
        'text-embedding-3-large': {
            'function': OpenAIEmbeddingFunction,
            'create_parms': {'model_name': 'text-embedding-3-large', 'api_key': config.env.get('kfOPENAI_API_KEY')},
            'read_parms': {'api_key': config.env.get('kfOPENAI_API_KEY')},
        },
        'text-embedding-ada-002': {
            'function': OpenAIEmbeddingFunction,
            'create_parms': {'model_name': 'text-embedding-ada-002', 'api_key': config.env.get('kfOPENAI_API_KEY')},
            'read_parms': {'api_key': config.env.get('kfOPENAI_API_KEY')},
        },
        'text-embedding-3-small': {
            'function': OpenAIEmbeddingFunction,
            'create_parms': {'model_name': 'text-embedding-3-small', 'api_key': config.env.get('kfOPENAI_API_KEY')},
            'read_parms': {'api_key': config.env.get('kfOPENAI_API_KEY')},
        },
    },
    'Google-GenerativeAI-Embeddings': {
        # gemini-embedding-exp-03-07
        'models/gemini-embedding-exp-03-07': {
            'function': GoogleGenerativeAiEmbeddingFunction,
            'create_parms': {'model_name': 'models/gemini-embedding-exp-03-07', 'api_key': config.env.get('kfGEMINI_API_KEY')},
            'read_parms': {'api_key': config.env.get('kfGEMINI_API_KEY')},
        },
        'models/text-embedding-004': {
            'function': GoogleGenerativeAiEmbeddingFunction,
            'create_parms': {'model_name': 'models/text-embedding-004', 'api_key': config.env.get('kfGEMINI_API_KEY')},
            'read_parms': {'api_key': config.env.get('kfGEMINI_API_KEY')},
        },
        'models/embedding-001': {
            'function': GoogleGenerativeAiEmbeddingFunction,
            'create_parms': {'model_name': 'models/embedding-001', 'api_key': config.env.get('kfGEMINI_API_KEY')},
            'read_parms': {'api_key': config.env.get('kfGEMINI_API_KEY')},
        },
    },
    'Ollama-Embeddings': {
        'nomic-embed-text': {
            'function': OllamaEmbeddingFunction,
            'create_parms': {'model_name': 'nomic-embed-text', 'url': 'http://localhost:11434/api/embeddings'},
            'read_parms': {},
        },
        'snowflake-arctic-embed2': {
            'function': OllamaEmbeddingFunction,
            'create_parms': {'model_name': 'snowflake-arctic-embed2', 'url': 'http://localhost:11434/api/embeddings'},
            'read_parms': {},
        },
        'mxbai-embed-large': {
            'function': OllamaEmbeddingFunction,
            'create_parms': {'model_name': 'mxbai-embed-large', 'url': 'http://localhost:11434/api/embeddings'},
            'read_parms': {},
        },
        # 'granite-embedding:278m': {
        #     'function': OllamaEmbeddingFunction,
        #     'create_parms': {'model_name': 'granite-embedding:278m', 'url': 'http://localhost:11434/api/embeddings'},
        #     'read_parms': {},
        # },
    },
    'Github-OpenAI-Embeddings': {
        'text-embedding-3-large': {
            'function': OpenAIEmbeddingFunction,
            'create_parms': {'model_name': 'text-embedding-3-large', 'api_key': config.env.get('kfGITHUB_TOKEN')},
            'read_parms': {'api_key': config.env.get('kfGITHUB_TOKEN')},
        },
        'text-embedding-ada-002': {
            'function': OpenAIEmbeddingFunction,
            'create_parms': {'model_name': 'text-embedding-ada-002', 'api_key': config.env.get('kfGITHUB_TOKEN')},
            'read_parms': {'api_key': config.env.get('kfGITHUB_TOKEN')},
        },
        'text-embedding-3-small': {
            'function': OpenAIEmbeddingFunction,
            'create_parms': {'model_name': 'text-embedding-3-small', 'api_key': config.env.get('kfGITHUB_TOKEN')},
            'read_parms': {'api_key': config.env.get('kfGITHUB_TOKEN')},
        },
    },
}


def _fix(s: str) -> str:
    return s.replace('\'', '\'\'')


class VSChroma(VSAPI):

    @staticmethod
    def embedding_types_list(embedding_type: str = None) -> list[str]:
        if embedding_type is None:
            return list(chroma_embedding_types.keys())
        else:
            return list(chroma_embedding_types[embedding_type].keys())

    @staticmethod
    def doc_loaders_list() -> list[str]:
        return list(lc_docloaders.docloaders.keys())

    @staticmethod
    def chunkers_list() -> list[str]:
        return list(lc_chunkers.chunkers.keys())

    class EmptyIngestError(Exception):
        pass

    class OllamaEmbeddingsError(Exception):
        pass

    def __init__(self, vs_type_name: str, vssettings: VSSettings, parms: dict[str, str]):
        super().__init__(vs_type_name, vssettings, parms)
        self._settings: VSChromaSettings = VSChromaSettings.from_settings(vssettings)
        self._client: chromadb.ClientAPI | None = None
        self.collection_name: str | None = None  # todo: get rid of this
        self._collection: chromadb.Collection | None = None  # todo: and this

    def warmup(self):
        self._build_clients()

    @staticmethod
    def create(vs_type_name: str, vssettings: VSSettings, parms: dict[str, str]) -> VSAPI:
        return VSChroma(vs_type_name, vssettings, parms)

    def settings(self) -> VSSettings:
        return self._settings

    def get_partial_collection(self, collection_name: str) -> Collection:
        """
        gets a PARTIALLY complete collection, has all metadata BUT can't be used to add data!
        :param collection_name:
        :return:
        """
        try:
            start = timeit.default_timer()
            collection = self._client.get_collection(name=collection_name)
            log.debug(f'{collection_name} metadata load {timeit.default_timer() - start: .1f}s')
            return collection
        except InvalidCollectionException:
            errmsg = f'bad collection name: {self.collection_name}'
            log.warning(errmsg)
            raise ValueError(errmsg)

    def get_collection(self, collection_name: str) -> Collection:
        """
        gets a COMPLETE collection, metadata + can be used to add data
        :param collection_name:
        :return:
        """
        try:
            # first figure out the embedding function safely in case e.g. we need a key for it
            start = timeit.default_timer()
            metadata = self._client.get_collection(name=collection_name).metadata
            log.debug(f'{collection_name} metadata load {timeit.default_timer() - start: .1f}s')
            if 'embedding_function_name' in metadata:
                ef_type: str = metadata['embedding_type'] if 'embedding_type' in metadata else 'unknown'
                ef_name: str = metadata['embedding_function_name'] if 'embedding_function_name' in metadata else 'unknown'
                ef_parms: str = metadata['embedding_function_parms'] if 'embedding_function_parms' in metadata else 'unknown'
                model_name = json.loads(ef_parms)['model_name']
                ef: chroma_api_types.EmbeddingFunction[chroma_api_types.Documents] = chroma_embedding_types[ef_type][model_name]['function']
                ef_parms: dict[str, str] = json.loads(metadata['embedding_function_parms'])
                ef_parms.update(chroma_embedding_types[ef_type][model_name]['read_parms'])  # adds e.g. a key

                start = timeit.default_timer()
                # noinspection PyTypeChecker
                full_collection = self._client.get_collection(
                    name=collection_name,
                    embedding_function=ef(**ef_parms)
                )
                log.debug(f'{collection_name} full load {timeit.default_timer() - start: .1f}s')
                return full_collection
            else:
                raise ValueError(f'collection {collection_name} has no embedding_function_name in metadata!')
        except InvalidCollectionException:
            errmsg = f'bad collection name: {self.collection_name}'
            log.warning(errmsg)
            raise ValueError(errmsg)

    def _build_clients(self):
        if self._client is not None:
            return

        log.info(f'building VS API for [{self._vs_type_name}]: {self.parms.get("CHROMA_HOST")=}, {self.parms.get("CHROMA_PORT")=}')

        self._client = chromadb.HttpClient(host=self.parms.get("CHROMA_HOST"), port=int(self.parms.get("CHROMA_PORT")))

    def list_collection_names(self) -> list[str]:
        self._build_clients()
        return [name for name in self._client.list_collections()]

    @validate_call()
    def embeddings_search(self, query: str, max_results: Annotated[int, Field(strict=True, ge=1)] = 10) -> VSAPI.SearchResponse:
        self._build_clients()

        # QueryResult is a TypedDict:
        # str->list (1 per query) of lists (1 per result): 'ids'->[[str]], 'distances'->[[float]], 'embeddings'-> None?, 'metadatas'->[[None?]]
        # ...'documents'->[[str]], 'uris'->None?, 'data'->None? 'included'->['distances', 'documents', 'metadatas']

        results: chroma_api_types.QueryResult = self._collection.query(
            query_texts=[query],  # chroma automatically creates embedding(s) for queries
            n_results=max_results,
            include=[chroma_api_types.IncludeEnum('documents'), chroma_api_types.IncludeEnum('metadatas'), chroma_api_types.IncludeEnum('distances'),
                     chroma_api_types.IncludeEnum('uris'), chroma_api_types.IncludeEnum('data')],
        )
        results_keys = results.keys()  # 'documents', 'metadatas', ...

        # chroma results are lists with one element per query, since we only have 1 query we only use element 0 from each list
        query_idx = 0
        # transform from dict-of-lists to more sensible list-of-dicts
        raw_results = []
        # for each result
        for i in range(0, len(results['ids'][query_idx])):  # get the result count for this query from a reliable key
            rdict = {}
            # add each of the keys to this result rdict, indexed by the result number i
            for k in results_keys:
                if k != 'included':  # we ignore the 'included' key, it just lists the keys that were included in the results
                    # noinspection PyTypedDict
                    rdict[k] = results[k][query_idx][i] if results[k] is not None and len(results[k]) > 0 and len(results[k][query_idx]) > i else None
            raw_results.append(rdict)

        return VSAPI.SearchResponse(
            results_text=[r['documents'] for r in raw_results],
            results_score=[r['distances'] for r in raw_results],
            results_raw=raw_results
        )

    @validate_call
    def search(self, query: str, max_results: int = 0, dense_weight: Annotated[float, Field(strict=True, ge=0.0, le=1.0)] = 0.5) -> VectorStoreResponse | None:
        self._build_clients()

        vs_results: list[VectorStoreResult] = []

        # embeddings search results (aka "dense")
        if dense_weight > 0.0:
            try:
                # todo: chroma (and others?) don't allow a max for embedded (dense) searches, so we use... 10 of course :)
                sresp: VSAPI.SearchResponse = self.embeddings_search(query, max_results=max_results if max_results > 0 else 10)
                for result_idx in range(0, len(sresp.results_raw)):
                    metrics = {
                        VSAPI.search_type_metric_name: VSAPI.search_type_embeddings,
                        'distance': sresp.results_raw[result_idx]['distances'],
                        'chunk metadata': sresp.results_raw[result_idx]['metadatas'],
                        'uris': sresp.results_raw[result_idx]['uris'],
                        'data': sresp.results_raw[result_idx]['data'],
                        'id': sresp.results_raw[result_idx]['ids'],
                    }
                    vs_results.append(VectorStoreResult(result_id=sresp.results_raw[result_idx]['ids'],
                                                        metrics=metrics,
                                                        content=sresp.results_raw[result_idx]['documents']))
            except (Exception,) as e:
                log.warning(f' embeddings_search error! {e}')
                raise e

        # full-text results (aka "sparse")
        if dense_weight < 1.0:
            try:
                # todo: fresh connection every time necessary?
                log.debug(f'connecting to sql: {config.sql_path}.{config.sql_chunks_table_name}')
                with sqlite3.connect(config.sql_path) as sql:
                    cursor = sql.cursor()

                    # FTS5 table full-text search using MATCH operator, plus bm25 (smaller=better match)
                    bm25_fragment = f"bm25({config.sql_chunks_fts5[self._settings.fts_type].table_name}, 0, 1, 0, 0)"
                    query = (f"select *, {bm25_fragment} bm25 "
                             f"from {config.sql_chunks_fts5[self._settings.fts_type].table_name} where content match '{query}' order by {bm25_fragment};")
                    log.debug(f'query {config.sql_chunks_table_name}: {query}')
                    cursor.execute(query)
                    colnames = [d[0] for d in cursor.description]
                    for row in cursor.fetchall():
                        rowdict = dict(zip(colnames, row))
                        metrics = {VSAPI.search_type_metric_name: VSAPI.search_type_fulltext}
                        metrics.update({e: rowdict[e] for e in rowdict if e not in ['content', 'id']})
                        vs_results.append(VectorStoreResult(
                            result_id=rowdict['id'],
                            metrics=metrics,
                            content=rowdict['content'],
                        ))
            except (Exception,) as e:
                log.warning(f'SQL error! {e}')
                raise e

        # # Combine the document IDs and remove duplicates
        # all_doc_ids = list(set(dense_doc_ids + sparse_doc_ids))
        #
        # # Create dictionaries to store the reciprocal ranks
        # dense_reciprocal_ranks = {doc_id: 0.0 for doc_id in all_doc_ids}
        # sparse_reciprocal_ranks = {doc_id: 0.0 for doc_id in all_doc_ids}
        #
        # # Step 2: Calculate the reciprocal rank for each document in dense and sparse search results.
        # for i, doc_id in enumerate(dense_doc_ids):
        #     dense_reciprocal_ranks[doc_id] = 1.0 / (i + 1)
        #
        # for i, doc_id in enumerate(sparse_doc_ids):
        #     sparse_reciprocal_ranks[doc_id] = 1.0 / (i + 1)
        #
        # # Step 3: Sum the reciprocal ranks for each document.
        # combined_reciprocal_ranks = {doc_id: 0.0 for doc_id in all_doc_ids}
        # for doc_id in all_doc_ids:
        #     combined_reciprocal_ranks[doc_id] = dense_weight * dense_reciprocal_ranks[doc_id] + sparse_weight * sparse_reciprocal_ranks[doc_id]
        #
        # # Step 4: Sort the documents based on their combined reciprocal rank scores.
        # sorted_doc_ids = sorted(all_doc_ids, key=lambda doc_id: combined_reciprocal_ranks[doc_id], reverse=True)
        #
        # # Step 5: Retrieve the documents based on the sorted document IDs.
        # sorted_docs = []
        # all_docs = dense_docs + sparse_docs
        # for doc_id in sorted_doc_ids:
        #     matching_docs = [doc for doc in all_docs if doc.metadata['id'] == doc_id]
        #     if matching_docs:
        #         doc = matching_docs[0]
        #         doc.metadata['score'] = combined_reciprocal_ranks[doc_id]
        #         doc.metadata['rank'] = sorted_doc_ids.index(doc_id) + 1
        #         if len(matching_docs) > 1:
        #             doc.metadata['retriever'] = 'both'
        #         elif doc in dense_docs:
        #             doc.metadata['retriever'] = 'dense'
        #         else:
        #             doc.metadata['retriever'] = 'sparse'
        #         sorted_docs.append(doc)
        #
        # # Step 7: Return the final ranked and sorted list, truncated by the top-k parameter
        # return sorted_docs[:k]

        return VectorStoreResponse(vs_results)

    def delete_collection(self, collection_name: str):
        self._build_clients()

        self._client.delete_collection(collection_name)
        self._collection = None

        log.debug(f'connecting to sql: {config.sql_path}.{config.sql_chunks_table_name}')
        sql = None
        try:
            # todo: fresh connection every time necessary?
            sql = sqlite3.connect(config.sql_path)
            cursor = sql.cursor()

            delete = f"delete from {config.sql_chunks_table_name} where collection ='{collection_name}';"
            log.debug(f'delete {config.sql_chunks_table_name}: {delete}')
            cursor.execute(delete)

            # NOTE: fts tables are chroma "external content" with delete triggers so no need for explicit deletes

            sql.commit()

        except (Exception,) as e:
            log.warning(f'SQL error! {e}')
            raise e
        finally:
            if sql is not None:
                sql.close()

    def count(self):
        """

        :return: the total number of embeddings in the collection
        """
        self._build_clients()
        self._collection.count()

    def peek(self, limit: int = 10):
        """

        :return: Get the first few results in the database up to limit
        """
        self._build_clients()
        results = self._collection.peek(limit=limit)

        # transform from dict-of-lists to more sensible list-of-dicts
        # NOTE: peek returns simple lists as the dict values (rather than the lists-of-lists that query returns)
        lod = []
        for i in range(0, len(results['ids'])):
            rdict = {}
            for k in results.keys():
                # noinspection PyTypedDict
                if k != 'included':  # the 'included' key is diff from the rest
                    # noinspection PyTypedDict
                    rdict[k] = results[k][i] if results[k] is not None else None
            lod.append(rdict)

        return lod

    def collection_dict(self) -> dict:
        self._build_clients()
        return self._collection.__dict__

    @staticmethod
    def filename_to_collname(file_name: str) -> str:
        """
        chroma collection name:
        (1) contains 3-63 characters,
        (2) starts and ends with an alphanumeric character,
        (3) otherwise contains only alphanumeric characters, underscores or hyphens (-) [NO SPACES!],
        (4) contains no two consecutive periods (..) and
        (5) is not a valid IPv4 address

        :param file_name:
        :return:
        """
        collection_name = file_name.replace(' ', '-')
        collection_name = collection_name.replace('..', '._')
        if len(collection_name) > 63:
            collection_name = collection_name[:63]
            log.warning(f'collection name too long, shortened')

        if collection_name != file_name:
            log.info(f'collection name [{file_name}] modified for chroma restrictions to [{collection_name}]')

        return collection_name

    def create_collection(self, name: str, fts_types: list[FTSType], embedding_type: str, subtype: str) -> Collection:
        """
        chroma collection name requirements:
        (1) contains 3-63 characters,
        (2) starts and ends with an alphanumeric character,
        (3) otherwise contains only alphanumeric characters, underscores or hyphens (-) [NO SPACES!],
        (4) contains no two consecutive periods (..) and
        (5) is not a valid IPv4 address
        (*) unique per... tenant? database? all?

        :param name:
        :param fts_types:
        :param embedding_type:
        :param subtype:
        """
        self._build_clients()

        # create the collection
        # todo: configure all this
        # recommended hnsw config and info from: https://stackoverflow.com/questions/78589963/insert-thousands-of-documents-into-a-chroma-db
        # metadata={
        #     "hnsw:space": "cosine",
        #     "hnsw:construction_ef": 600,
        #     "hnsw:search_ef": 1000,
        #     "hnsw:M": 60
        # },
        # todo: CollectionConfiguration will eventually be implemented: https://github.com/chroma-core/chroma/pull/2495
        embedding_function_info = chroma_embedding_types[embedding_type][subtype]  # default: 'all-MiniLM-L6-v2'
        #  x: CollectionConfiguration = CollectionConfiguration(hnsw_configuration=HNSWConfiguration(space='cosine'))
        collection: Collection = self._client.create_collection(
            name=name,
            metadata={
                'fts_types': f'{json.dumps([f.name for f in fts_types])}',

                'embedding_type': embedding_type,
                'embedding_function_name': embedding_function_info['function'].__name__,
                'embedding_function_parms': json.dumps(self.filter_metadata(embedding_function_info['create_parms']), ensure_ascii=False),

                'hnsw:space': 'cosine',  # default l2
                'hnsw:construction_ef': 600,  # default 100
                'hnsw:search_ef': 1000,  # default 10
                'hnsw:M': 60,  # default 16

                'chroma_version': self._client.get_version(),
                'created': config.now_datetime()
            },
            embedding_function=embedding_function_info['function'](**embedding_function_info['create_parms']),
            data_loader=None,
            get_or_create=False
        )

        return collection

    @staticmethod
    def fix_metadata_for_modify(md: dict[str, any]) -> dict[str, any]:
        """
        this is necessary b/c the hnsw parameters are passed as metadata currently and some (e.g. "hnsw:space") CAN'T BE CHANGED
        see: https://github.com/chroma-core/chroma/issues/2515
        (the suggestion to use client.get_or_create_collection instead of collection.modify didn't work!
        :param md:
        :return:
        """
        retval: dict[str, any] = {}
        for k, v in md.items():
            retval[k if not k.startswith('hnsw') else f'org-{k}'] = v
        return retval

    # borrowed from langchain_community.vectorstores.utils.filter_complex_metadata
    md_allowed_types: tuple[type, ...] = (str, bool, int, float)

    # todo: config this?
    md_redacted_keys: list[str] = ['api_key', '_api_key']

    def filter_metadata_docs(self, documents: list[Document], org_filename: str) -> list[Document]:
        """
        filters and does some *light* processing on docs from chunker:
        - removes docs with metadata that's not one of the allowed types
        - removes docs with len(page_content) == 0
        - redacts any secret keys
        - replaces any 'source' metadata with the original filename instead of the useless temporary filename
        :param documents:
        :param org_filename:
        :return:
        """
        good_docs: list[Document] = []
        for doc in documents:
            if len(doc.page_content) > 0:  # some chunkers can't handle 0-length docs (and why bother keeping them anyway?)
                doc_md = {}
                for key, value in doc.metadata.items():
                    if not isinstance(value, self.md_allowed_types):
                        continue
                    if key in self.md_redacted_keys:
                        value = config.redact(value)

                    # fix 'source' in chunks metadata since it references a tmp file which is pointless
                    if key == 'source':
                        value = org_filename

                    doc_md[key] = value

                doc.metadata = doc_md
                good_docs.append(doc)
            else:
                log.debug(f'{self.collection_name}:{org_filename}: filtered 0-length chunk ({doc.metadata})')

        return good_docs

    def filter_metadata(self, metadata: dict[str, any]) -> dict[str, any]:
        """
        removes values that aren't in allowed_types and
        :param metadata:
        :return:
        """
        retval: dict[str, any] = {}
        for key, value in metadata.items():
            if not isinstance(value, self.md_allowed_types):
                continue
            if key in self.md_redacted_keys:
                value = config.redact(value)
            retval[key] = value
        return retval

    def ingest(self, collection: Collection, server_file_path: str, org_filename: str, docloader_type: str, chunker_type: str) -> Collection | None:
        if docloader_type in lc_docloaders.docloaders:
            log.debug(f'loading {org_filename} for {collection.name} with {docloader_type}')
            docs: list[Document] = lc_docloaders.docloaders[docloader_type]['function'](file_path=server_file_path).load()
        else:
            raise ValueError(f'unknown doc loader/type [{docloader_type}]')

        # chunking
        if chunker_type in lc_chunkers.chunkers:
            log.debug(f'chunking {len(docs)} documents for {collection.name} with {chunker_type}')
            chunker_func = lc_chunkers.chunkers[chunker_type]['function']
            chunker_args = lc_chunkers.chunkers[chunker_type]['args']
            chunks: list[Document] = self.filter_metadata_docs(chunker_func(**chunker_args).split_documents(docs), org_filename)
        else:
            raise ValueError(f'unknown chunker type [{chunker_type}]')

        if len(chunks) == 0:
            raise VSChroma.EmptyIngestError(f'no usable chunks (empty file?)')

        # add file-level metadata
        now = config.now_datetime()
        collection.metadata[f'file:{org_filename}:upload time'] = now
        collection.metadata[f'file:{org_filename}:docloader_type'] = f'{docloader_type}: {lc_docloaders.docloaders[docloader_type]['function'].__name__}/{lc_docloaders.docloaders[docloader_type]['filetypes']}'
        collection.metadata[f'file:{org_filename}:doc/chunk counts'] = f'{len(docs)}/{len(chunks)}'
        collection.metadata[f'file:{org_filename}:chunker_type'] = f'{chunker_type}: {lc_chunkers.chunkers[chunker_type]['function'].__name__}'
        for key, value in chunker_args.items():
            if isinstance(value, OpenAIEmbeddings):  # if it's an oai function, pull out known fields for metadata
                oaie: OpenAIEmbeddings = value
                collection.metadata[f'file:{org_filename}:chunker.model.embeddings'] = OpenAIEmbeddings.__name__
                collection.metadata[f'file:{org_filename}:chunker.model'] = oaie.model
                #  collection.metadata[f'file:{org_filename}:chunker.dimensions'] = oaie.dimensions # always None!?
            else:
                collection.metadata[f'file:{org_filename}:chunker.{key}'] = value
        collection.modify(metadata=self.fix_metadata_for_modify(self.filter_metadata(collection.metadata)))

        # get needed info from collection-level metadata
        fts_types = [FTSType[ftype] for ftype in json.loads(collection.metadata.get('fts_types'))]

        #  add chunks + ids + metadata to the collection
        log.debug(f'adding embeddings for {len(chunks)} chunks [{chunker_type}] to collection {collection.name}')
        try:
            chunks_content = [c.page_content for c in chunks]
            ids = [f'{org_filename}-{i}' for i in range(0, len(chunks))]  # use name:count for chunk-ids
            metadata = [c.metadata for c in chunks]

            collection.add(documents=chunks_content, ids=ids, metadatas=metadata)

            sql = None
            try:
                # todo: fresh connection every time
                log.debug(f'connecting to sql: {config.sql_path}.{config.sql_chunks_table_name}')
                sql = sqlite3.connect(config.sql_path)
                cursor = sql.cursor()

                # create tables needed for full-text search
                # the content table
                log.debug(f'create {config.sql_chunks_table_name}: {config.sql_chunks_create}')
                cursor.execute(config.sql_chunks_create)

                # need to combine statements for each trigger type (i/d/u) from every fts_type
                insert_trigger_stmts = []
                delete_trigger_stmts = []
                update_trigger_stmts = []
                for fts_type in fts_types:
                    # full-text search tables/collections
                    log.debug(f'create fts5 {config.sql_chunks_fts5[fts_type].table_name}: {config.sql_chunks_fts5[fts_type].create}')
                    insert_trigger_stmts.append(config.sql_chunks_fts5[fts_type].insert_trigger)
                    delete_trigger_stmts.append(config.sql_chunks_fts5[fts_type].delete_trigger)
                    update_trigger_stmts.append(config.sql_chunks_fts5[fts_type].update_trigger)
                    cursor.execute(config.sql_chunks_fts5[fts_type].create)
                itrigger = f"{config.sql_chunks_insert_trigger_create} begin\n{'\n'.join(insert_trigger_stmts)}\nend;"
                log.debug(f'add insert trigger: {itrigger}')
                cursor.execute(itrigger)
                dtrigger = f"{config.sql_chunks_delete_trigger_create} begin\n{'\n'.join(delete_trigger_stmts)}\nend;"
                log.debug(f'add delete trigger: {dtrigger}')
                cursor.execute(dtrigger)
                utrigger = f"{config.sql_chunks_update_trigger_create} begin\n{'\n'.join(update_trigger_stmts)}\nend;"
                log.debug(f'add update trigger: {utrigger}')
                cursor.execute(utrigger)

                # insert the chunks
                log.debug(f'inserting {len(chunks)} chunks into {config.sql_chunks_table_name}')
                for i, chunk_content in enumerate(chunks_content):
                    insert = f"insert into {config.sql_chunks_table_name} values ('{_fix(collection.name)}', '{_fix(chunk_content)}', '{_fix(ids[i])}', '{_fix(str(metadata[i]))}', NULL)"
                    # log.debug(f'insert: {insert.replace("\n", "[\\n]")}')
                    cursor.execute(insert)

                sql.commit()
            except (Exception,) as e:
                log.warning(f'SQL error! {e}')
                raise e
            finally:
                if sql is not None:
                    sql.close()

        except (Exception,) as e:
            collection_md = collection.metadata
            e_type: str = collection_md['embedding_type'] if 'embedding_type' in collection_md else 'unknown'
            ef_name: str = collection_md['embedding_function_name'] if 'embedding_function_name' in collection_md else 'unknown'
            errmsg = f'Error adding embeddings to {collection.name} function:{ef_name} type:{e_type}: {e.__class__.__name__}: {e}'
            log.warning(errmsg)
            if 'OLLAMA' in e_type.lower():
                raise VSChroma.OllamaEmbeddingsError(errmsg + ' (is model loaded in ollama?)')
            else:
                raise e

        return collection

    def switch_collection(self, new_collection_name: str) -> None:
        log.info(f'switching collection to [{new_collection_name}]')
        self._build_clients()
        self.collection_name = new_collection_name
        self._collection: Collection = self.get_collection(self.collection_name)
