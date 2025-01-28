import json
import logging
import timeit

import chromadb
import chromadb.api.types as chroma_api_types
from chromadb.api.models.Collection import Collection
from chromadb.errors import InvalidCollectionException
from chromadb.utils.embedding_functions.
from chromadb.utils.embedding_functions.google_embedding_function import GoogleGenerativeAiEmbeddingFunction
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import SentenceTransformerEmbeddingFunction
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import config
import logstuff
from chatexchanges import VectorStoreResponse, VectorStoreResult
from lc_chunkers import chunkers
from lc_docloaders import docloaders
from vsapi import VSAPI

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class VSChroma(VSAPI):

    def __init__(self, api_type_name: str, parms: dict[str, str]):
        super().__init__(api_type_name, parms)
        self._client: chromadb.ClientAPI | None = None
        self.collection_name: str | None = None  # todo: get rid of this
        self._collection: chromadb.Collection | None = None  # todo: and this

    # function-name-string : {model-name : {function, create_parms: {model_name}, read_parms: {}}}
    embedding_types: dict[str, dict[str, dict[str, any]]] = {
        SentenceTransformerEmbeddingFunction.__name__: {
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
        OpenAIEmbeddingFunction.__name__: {
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
        GoogleGenerativeAiEmbeddingFunction.__name__: {
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
        OllamaEmbeddingFunction.__name__: {
            'abc': {
                'function': OllamaEmbeddingFunction,
                'create_parms': {},
                'read_parms': {},
            }
        },
    }

    def warmup(self):
        self._build_clients()

    @staticmethod
    def create(api_type_name: str, parms: dict[str, str]):
        return VSChroma(api_type_name, parms)

    def get_collection_metadata(self, collection_name: str) -> Collection:
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
                ef_name: str = metadata['embedding_function_name']
                ef_parms: str = metadata['embedding_function_parms']
                model_name = json.loads(ef_parms)['model_name']
                ef: chroma_api_types.EmbeddingFunction[chroma_api_types.Documents] = self.embedding_types[ef_name][model_name]['function']
                ef_parms: dict[str, str] = json.loads(metadata['embedding_function_parms'])
                ef_parms.update(self.embedding_types[ef_name][model_name]['read_parms'])  # adds e.g. a key

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

        log.info(f'building VS API for [{self._api_type_name}]: {self.parms.get("CHROMA_HOST")=}, {self.parms.get("CHROMA_PORT")=}')

        self._client = chromadb.HttpClient(host=self.parms.get("CHROMA_HOST"), port=int(self.parms.get("CHROMA_PORT")))

    def list_index_names(self) -> list[str]:
        self._build_clients()
        return [name for name in self._client.list_collections()]

    def raw_search(self, prompt: str, howmany: int) -> VSAPI.SearchResponse:
        self._build_clients()

        # dict str->list (1 per prompt) of lists (1 per result): 'ids'->[[str]], 'distances'->[[float]], 'embeddings'-> None?, 'metadatas'->[[None?]]
        # ...'documents'->[[str]], 'uris'->None?, 'data'->None? 'included'->['distances', 'documents', 'metadatas']
        results: dict = self._collection.query(
            query_texts=[prompt],  # chroma will automatically creates embedding(s) for these
            n_results=howmany,
            include=[chroma_api_types.IncludeEnum('documents'), chroma_api_types.IncludeEnum('metadatas'), chroma_api_types.IncludeEnum('distances'),
                     chroma_api_types.IncludeEnum('uris'), chroma_api_types.IncludeEnum('data')],
        )

        # chroma results are lists with one element per prompt, since we only have 1 prompt we only use element 0 from each list
        # transform from dict-of-lists to more sensible list-of-dicts
        raw_results = []
        # for each expected result
        for i in range(0, howmany):
            rdict = {}
            # for each of the chroma result keys
            for k in results:
                if k != 'included':  # we ignore the 'included' key
                    rdict[k] = results[k][0][i] if results[k] is not None and len(results[k]) > 0 and len(results[k][0]) > i else None
            raw_results.append(rdict)

        return VSAPI.SearchResponse(
            results_text=[r['documents'] for r in raw_results],
            results_score=[r['distances'] for r in raw_results],
            results_raw=raw_results
        )

    def search(self, prompt: str, howmany: int, source_name: str, source_type: str) -> VectorStoreResponse:
        sresp: VSAPI.SearchResponse = self.raw_search(prompt, howmany)

        vs_results: list[VectorStoreResult] = []
        for result_idx in range(0, len(sresp.results_raw)):
            metrics = {
                'distance': sresp.results_raw[result_idx]['distances'],
                'chunk metadata': sresp.results_raw[result_idx]['metadatas'],
                'uris': sresp.results_raw[result_idx]['uris'],
                'data': sresp.results_raw[result_idx]['data'],
                'id': sresp.results_raw[result_idx]['ids'],
            }
            vs_results.append(VectorStoreResult(sresp.results_raw[result_idx]['ids'], metrics,
                                                sresp.results_raw[result_idx]['documents']))
        return VectorStoreResponse(vs_results, source_name=source_name, source_type=source_type)

    def delete_index(self, index_name: str):
        self._build_clients()
        self._client.delete_collection(index_name)

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
    def compute_collection_name(file_name: str) -> str:
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

    def create_collection(self, name: str, embedding_type: str, subtype: str) -> Collection:
        """
        chroma collection name requirements:
        (1) contains 3-63 characters,
        (2) starts and ends with an alphanumeric character,
        (3) otherwise contains only alphanumeric characters, underscores or hyphens (-) [NO SPACES!],
        (4) contains no two consecutive periods (..) and
        (5) is not a valid IPv4 address
        (*) unique per... tenant? database? all?

        :param name:
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
        # default hnsw: {'space': 'l2', 'construction_ef': 100, 'search_ef': 10, 'num_threads': 22, 'M': 16, 'resize_factor': 1.2, 'batch_size': 100, 'sync_threshold': 1000, '_type': 'HNSWConfigurationInternal'}
        # todo: CollectionConfiguration will eventually be implemented: https://github.com/chroma-core/chroma/pull/2495
        embedding_function_info = self.embedding_types[embedding_type][subtype]  # default: 'all-MiniLM-L6-v2'
        #  x: CollectionConfiguration = CollectionConfiguration(hnsw_configuration=HNSWConfiguration(space='cosine'))
        collection: Collection = self._client.create_collection(
            name=name,
            metadata={
                'embedding_function_name': embedding_function_info['function'].__name__,
                'embedding_function_parms': json.dumps(self.filter_metadata(embedding_function_info['create_parms']), ensure_ascii=False),

                'hnsw:space': 'l2',  # default l2
                'hnsw:construction_ef': 500,  # default 100
                'hnsw:search_ef': 500,  # default 10
                'hnsw:M': 40,  # default 16

                'chroma_version': self._client.get_version(),
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
    md_redacted_keys: list[str] = ['api_key', '_api_key']

    def filter_metadata_docs(self, documents: list[Document], org_filename: str) -> list[Document]:
        retval: list[Document] = []
        for doc in documents:
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
            retval.append(doc)

        return retval

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

    def ingest(self, collection: Collection, server_file_path: str, org_filename: str, doc_type: str, chunker_type: str, chunker_args: dict[str, any]) -> Collection:
        if doc_type in docloaders:
            log.debug(f'loading {org_filename} for {collection.name} with {doc_type}')
            docs: list[Document] = docloaders[doc_type]['function'](file_path=server_file_path).load()
        else:
            raise ValueError(f'unknown doc loader/type [{doc_type}]')

        # chunking
        if chunker_type in chunkers:
            log.debug(f'chunking {len(docs)} documents for {collection.name} with {chunker_type} args: {chunker_args}')
            chunker_func = chunkers[chunker_type]['function']
            chunks = self.filter_metadata_docs(chunker_func(**chunker_args).split_documents(docs), org_filename)
        else:
            raise ValueError(f'unknown chunker type [{chunker_type}]')

        # update collection metadata
        now = config.now_datetime()
        collection.metadata[f'file:{org_filename}'] = now
        collection.metadata[f'file:{org_filename}.doc_type'] = doc_type
        collection.metadata[f'file:{org_filename}.chunker_type'] = chunker_type
        for key, value in chunker_args.items():
            if isinstance(value, OpenAIEmbeddings):  # if it's an oai function, pull out known fields for metadata
                oaie: OpenAIEmbeddings = value
                collection.metadata[f'file:{org_filename}.chunker.model.embeddings'] = OpenAIEmbeddings.__name__
                collection.metadata[f'file:{org_filename}.chunker.model'] = oaie.model
                #  collection.metadata[f'file:{org_filename}.chunker.dimensions'] = oaie.dimensions # always None!?
            else:
                collection.metadata[f'file:{org_filename}.chunker.{key}'] = value
        collection.modify(metadata=self.fix_metadata_for_modify(self.filter_metadata(collection.metadata)))

        #  add documents + ids + metadata to the collection
        log.debug(f'adding embeddings for {len(chunks)} [{chunker_type}] chunks to collection {collection.name}')
        collection.add(documents=[c.page_content for c in chunks],
                       ids=[f'{org_filename}-{i}' for i in range(0, len(chunks))],  # use name:count for chunk-ids
                       metadatas=[c.metadata for c in chunks],
                       )

        return collection

    def switch_index(self, new_index_name: str) -> None:
        log.info(f'switching index to [{new_index_name}]')
        self._build_clients()
        self.collection_name = new_index_name
        self._collection: Collection = self.get_collection(self.collection_name)
