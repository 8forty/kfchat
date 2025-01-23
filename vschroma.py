import json
import logging

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import IncludeEnum, EmbeddingFunction, Documents, Document
from chromadb.errors import InvalidCollectionException
from chromadb.utils.embedding_functions.google_embedding_function import GoogleVertexEmbeddingFunction, GoogleGenerativeAiEmbeddingFunction
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import SentenceTransformerEmbeddingFunction
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import logstuff
import chunkers
from chatexchanges import VectorStoreResponse, VectorStoreResult
from vsapi import VSAPI

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class VSChroma(VSAPI):

    def __init__(self, api_type_name: str, parms: dict[str, str]):
        super().__init__(api_type_name, parms)
        self._client: chromadb.ClientAPI | None = None
        self.collection_name: str | None = None  # todo: get rid of this
        self._collection: chromadb.Collection | None = None  # todo: and this

    embedding_functions: dict[str, EmbeddingFunction[Documents]] = {
        SentenceTransformerEmbeddingFunction.__name__: SentenceTransformerEmbeddingFunction,
        OpenAIEmbeddingFunction.__name__: OpenAIEmbeddingFunction,
        GoogleGenerativeAiEmbeddingFunction.__name__: GoogleGenerativeAiEmbeddingFunction,
        OllamaEmbeddingFunction.__name__: OllamaEmbeddingFunction
    }

    embedding_types_data: dict[str, any] = {
        'ST/all-MiniLM-L6-v2': {'function': SentenceTransformerEmbeddingFunction, 'parms': {'model_name': 'all-MiniLM-L6-v2'}},
        'ST/all-mpnet-base-v2': {'function': SentenceTransformerEmbeddingFunction, 'parms': {'model_name': 'all-mpnet-base-v2'}},
        'OpenAI/text-embedding-3-large': {'function': OpenAIEmbeddingFunction, 'parms': {'model_name': 'text-embedding-3-large',
                                                                                         'api_key': config.env.get('OPENAI_API_KEY'),  # todo: fix this!
                                                                                         }},
        'OpenAI/text-embedding-ada-002': {'function': OpenAIEmbeddingFunction, 'parms': {'model_name': 'text-embedding-ada-002'}},
        'OpenAI/text-embedding-3-small': {'function': OpenAIEmbeddingFunction, 'parms': {'model_name': 'text-embedding-3-small'}},
        'ST/Google': {'function': GoogleGenerativeAiEmbeddingFunction, 'parms': {}},
        'ST/Ollama': {'function': OllamaEmbeddingFunction, 'parms': {}},
    }

    def warmup(self):
        self._build_clients()

    @staticmethod
    def create(api_type_name: str, parms: dict[str, str]):
        return VSChroma(api_type_name, parms)

    def get_collection(self, collection_name: str) -> Collection:
        try:
            # first figure out the embedding function
            metadata = self._client.get_collection(name=collection_name).metadata
            if 'embedding_function_name' in metadata:
                embedding_function_name = metadata['embedding_function_name']
                ef: EmbeddingFunction[Documents] = self.embedding_functions[embedding_function_name]
                embedding_function_parms: dict[str, str] = json.loads(metadata['embedding_function_parms'])

                return self._client.get_collection(
                    name=collection_name,
                    embedding_function=ef(**embedding_function_parms)
                )
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
            include=[IncludeEnum('documents'), IncludeEnum('metadatas'), IncludeEnum('distances'), IncludeEnum('uris'), IncludeEnum('data')],
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

    def search(self, prompt: str, howmany: int) -> VectorStoreResponse:
        sresp: VSAPI.SearchResponse = self.raw_search(prompt, howmany)

        vs_results: list[VectorStoreResult] = []
        for result_idx in range(0, len(sresp.results_raw)):
            metrics = {
                'distance': sresp.results_raw[result_idx]['distances'],
                'metadata': sresp.results_raw[result_idx]['metadatas'],
                'uris': sresp.results_raw[result_idx]['uris'],
                'data': sresp.results_raw[result_idx]['data'],
                'id': sresp.results_raw[result_idx]['ids'],
            }
            vs_results.append(VectorStoreResult(sresp.results_raw[result_idx]['ids'], metrics,
                                                sresp.results_raw[result_idx]['documents']))
        return VectorStoreResponse(vs_results)

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

    def create_collection(self, name: str, embedding_type: str) -> Collection:
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
        """
        self._build_clients()

        # create the collection
        # todo: configure all this
        # todo: recommended hnsw config from:
        # metadata={
        #     "hnsw:space": "cosine",
        #     "hnsw:construction_ef": 600,
        #     "hnsw:search_ef": 1000,
        #     "hnsw:M": 60
        # },
        # default hnsw: {'space': 'l2', 'construction_ef': 100, 'search_ef': 10, 'num_threads': 22, 'M': 16, 'resize_factor': 1.2, 'batch_size': 100, 'sync_threshold': 1000, '_type': 'HNSWConfigurationInternal'}
        # todo: CollectionConfiguration will eventually be implemented: https://github.com/chroma-core/chroma/pull/2495
        embedding_function_info = self.embedding_types_data[embedding_type]  # default: 'all-MiniLM-L6-v2'
        #  x: CollectionConfiguration = CollectionConfiguration(hnsw_configuration=HNSWConfiguration(space='cosine'))
        collection: Collection = self._client.create_collection(
            name=name,
            metadata={
                # 'chunk_method': f'{VSChroma.__name__}.{self.ingest_pdf_text_splitter.__name__}',
                # 'original_filename:': f'{file_name}',
                # 'path': file_path,
                # 'chunk_size': chunk_size,
                # 'chunk_overlap': chunk_overlap,

                'embedding_function_name': embedding_function_info['function'].__name__,
                # todo: the key!?!? 'embedding_function_parms': json.dumps(config.redact_parms(embedding_function_info['parms']), ensure_ascii=False),
                'embedding_function_parms': json.dumps(embedding_function_info['parms'], ensure_ascii=False),

                'hnsw:space': 'l2',  # default l2
                'hnsw:construction_ef': 500,  # default 100
                'hnsw:search_ef': 500,  # default 10
                'hnsw:M': 40,  # default 16

                'chroma_version': self._client.get_version(),
            },
            embedding_function=embedding_function_info['function'](**embedding_function_info['parms']),
            data_loader=None,
            get_or_create=False
        )

        return collection

    def fix_metadata_for_modify(self, md: dict[str, any]) -> dict[str, any]:
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

    def ingest(self, collection: Collection, server_file_path: str, org_filename: str, doc_type: str, chunker_type: str, chunker_args: dict[str, any]) -> Collection:
        if doc_type == 'pypdf':
            # load the PDF into LC Document's
            docs: list[Document] = PyPDFLoader(file_path=server_file_path).load()
        else:
            raise ValueError(f'unknown document type [{doc_type}]')

        if chunker_type == 'rcts':
            # split into chunks
            chunks = RecursiveCharacterTextSplitter(**chunker_args).split_documents(docs)

            # remove complex metadata not supported by ChromaDB, pull out the the content as a str
            chunks = [c.page_content for c in filter_complex_metadata(chunks)]

        else:
            raise ValueError(f'unknown chunker type [{chunker_type}]')

        log.debug(f'adding embeddings for {len(chunks)} chunks to {collection.name}')
        now = config.now_datetime()
        collection.metadata[f'file:{org_filename}'] = now
        collection.metadata[f'file:{org_filename}.doc_type'] = doc_type
        collection.metadata[f'file:{org_filename}.chunker_type'] = chunker_type
        collection.modify(metadata=self.fix_metadata_for_modify(collection.metadata))

        #  collection.add(documents=chunks, ids=[str(uuid.uuid4()) for _ in range(0, len(chunks))])  # use random uuids for chunk-ids
        collection.add(documents=chunks,
                       ids=[f'{org_filename}-{i}' for i in range(0, len(chunks))],  # use name:count for chunk-ids
                       )

        return collection

    def switch_index(self, new_index_name: str) -> None:
        log.info(f'switching index to [{new_index_name}]')
        self._build_clients()
        self.collection_name = new_index_name
        self._collection: Collection = self.get_collection(self.collection_name)
