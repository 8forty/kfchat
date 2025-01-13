import logging
import uuid

import chromadb
from chromadb.api.types import IncludeEnum
from chromadb.errors import InvalidCollectionException
from chromadb.types import Collection
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import SentenceTransformerEmbeddingFunction

import logstuff
import pdf_chunkers
from vsapi import VSAPI

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class VSChroma(VSAPI):

    def __init__(self, api_type_name: str, parms: dict[str, str]):
        super().__init__(api_type_name, parms)
        self._client: chromadb.ClientAPI | None = None
        self.embedding_model_name: str = parms.get("CHROMA_EMBEDDING_MODEL")
        self.collection_name: str | None = None
        self._collection: chromadb.Collection | None = None

    def warmup(self):
        self._build_clients()

    @staticmethod
    def create(api_type_name: str, parms: dict[str, str]):
        return VSChroma(api_type_name, parms)

    def get_collection(self, collection_name: str) -> Collection:
        try:
            # todo: assuming SentenceTransformerEmbeddingFunction here
            return self._client.get_collection(
                name=collection_name,
                embedding_function=SentenceTransformerEmbeddingFunction(model_name=self.embedding_model_name),  # default: 'all-MiniLM-L6-v2'
                data_loader=None,
            )
        except InvalidCollectionException:
            errmsg = f'bad collection name: {self.collection_name}'
            log.warning(errmsg)
            raise ValueError(errmsg)

    def _build_clients(self):
        if self._client is not None:
            return

        log.info(f'building VS API for [{self._api_type_name}]: {self.parms.get("CHROMA_HOST")=}, {self.parms.get("CHROMA_PORT")=}, '
                 f'{self.parms.get("CHROMA_EMBEDDING_MODEL")=}, {self.parms.get("CHROMA_COLLECTION")=} ')
        self._client = chromadb.HttpClient(host=self.parms.get("CHROMA_HOST"), port=int(self.parms.get("CHROMA_PORT")))

    def list_index_names(self) -> list[str]:
        self._build_clients()
        return [c.name for c in self._client.list_collections()]

    def search(self, prompt: str, howmany: int) -> VSAPI.SearchResponse:
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
        for i in range(0, howmany):
            rdict = {}
            for k in results:
                if k != 'included':  # the 'included' key is diff from the rest
                    rdict[k] = results[k][0][i] if results[k] is not None else None
            raw_results.append(rdict)

        return VSAPI.SearchResponse(
            results_text=[r['documents'] for r in raw_results],
            results_score=[r['distances'] for r in raw_results],
            results_raw=raw_results
        )

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

    def ingest_pdf(self, pdf_file_path: str, pdf_name: str) -> Collection:
        self._build_clients()

        collection: Collection | None = None

        # todo: configure this
        chunks = pdf_chunkers.chunk_recursive_character_text_splitter(server_pdf_path=pdf_file_path, chunk_size=1000, chunk_overlap=200)
        if len(chunks) == 0:
            log.warning(f'no chunks found in {pdf_name} ({pdf_file_path})!')
        else:
            # chroma collection name:
            # (1) contains 3-63 characters,
            # (2) starts and ends with an alphanumeric character,
            # (3) otherwise contains only alphanumeric characters, underscores or hyphens (-) [NO SPACES!],
            # (4) contains no two consecutive periods (..) and
            # (5) is not a valid IPv4 address
            collection_name = pdf_name.replace(' ', '-')
            collection_name = collection_name.replace('..', '._')
            if len(collection_name) > 63:
                collection_name = collection_name[:63]
                log.warning(f'collection name too long, shortened')

            if collection_name != pdf_name:
                log.info(f'collection name [{pdf_name}] modified for chroma restrictions to [{collection_name}]')

            # create the collection
            # todo: configure this
            collection = self._client.create_collection(
                name=collection_name,
                configuration=None,
                metadata=None,
                embedding_function=SentenceTransformerEmbeddingFunction(model_name=self.embedding_model_name),  # default: 'all-MiniLM-L6-v2'
                data_loader=None,
                get_or_create=False
            )

            # create Chroma vectorstore from the chunks, use random uuids for chunk-ids
            log.debug(f'adding {len(chunks)} chunks to {collection.name}')
            collection.add(documents=chunks, ids=[str(uuid.uuid4()) for _ in range(0, len(chunks))])

        return collection

    def change_index(self, new_index_name: str) -> None:
        log.info(f'changing index to [{new_index_name}]')
        self._build_clients()
        self.collection_name = new_index_name
        self._collection: Collection = self.get_collection(self.collection_name)
