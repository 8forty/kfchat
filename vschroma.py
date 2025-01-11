import logging

import chromadb
from chromadb.api.types import IncludeEnum
from chromadb.errors import InvalidCollectionException
from chromadb.types import Collection
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import SentenceTransformerEmbeddingFunction

import logstuff
from vsapi import VSAPI

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class VSChroma(VSAPI):

    def __init__(self, api_type_name: str, index_name: str, parms: dict[str, str]):
        super().__init__(api_type_name, index_name, parms)
        self._client: chromadb.ClientAPI | None = None
        self.collection_name: str = parms.get("CHROMA_COLLECTION")
        self.embedding_model_name: str = parms.get("CHROMA_EMBEDDING_MODEL")

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
        try:
            # todo: assuming SentenceTransformerEmbeddingFunction here
            collection: Collection = self._client.get_collection(
                name=self.collection_name,
                embedding_function=SentenceTransformerEmbeddingFunction(model_name=self.embedding_model_name),  # default: 'all-MiniLM-L6-v2'
                data_loader=None,
            )
        except InvalidCollectionException:
            errmsg = f'bad collection name: {self.collection_name}'
            log.warning(errmsg)
            raise ValueError(errmsg)

        # dict str->list (1 per prompt) of lists (1 per result): 'ids'->[[str]], 'distances'->[[float]], 'embeddings'-> None?, 'metadatas'->[[None?]]
        # ...'documents'->[[str]], 'uris'->None?, 'data'->None? 'included'->['distances', 'documents', 'metadatas']
        results: dict = collection.query(
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
                rdict[k] = results[k][0][i] if results[k] is not None else None
            raw_results.append(rdict)

        return VSAPI.SearchResponse(
            results_text=[r['documents'] for r in raw_results],
            results_score=[r['distances'] for r in raw_results],
            results_raw=raw_results
        )

    @staticmethod
    def create(api_type_name: str, index_name: str, parms: dict[str, str]):
        return VSChroma(api_type_name, index_name, parms)
