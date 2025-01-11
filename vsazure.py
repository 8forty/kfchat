import logging

import openai
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents import SearchItemPaged
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery

import logstuff
from vsapi import VSAPI
from config import redact

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class VSAzure(VSAPI):

    def __init__(self, api_type_name: str, index_name: str, parms: dict[str, str]):
        super().__init__(api_type_name, index_name, parms)
        self._aoai_client: openai.AzureOpenAI | None = None
        self._aais_client: SearchClient | None = None
        self.deployment: str = parms.get("AZURE_AI_SEARCH_EMBEDDING_DEPLOYMENT")
        self.embedding_dimensions: int = int(parms.get("AZURE_AI_SEARCH_EMBEDDING_DIMENSIONS"))

    def _build_clients(self):
        if self._aoai_client is not None:
            return

        # token_provider = azure.identity.get_bearer_token_provider(
        #     azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        # )
        # client = openai.AzureOpenAI(
        #     api_version=self.parms.get("AZURE_OPENAI_API_VERSION"),
        #     azure_endpoint=self.parms.get("AZURE_OPENAI_ENDPOINT"),
        #     azure_ad_token_provider=token_provider,
        # )

        log.info(f'building VS API for [{self._api_type_name}]: {self.parms.get("AZURE_OPENAI_ENDPOINT")=}, '
                 f'AZURE_OPENAI_API_KEY={redact(self.parms.get("AZURE_OPENAI_API_KEY"))}, '
                 f'{self.parms.get("AZURE_OPENAI_API_VERSION")=}, {self.parms.get("AZURE_AI_SEARCH_ENDPOINT")=}, '
                 f'AZURE_AI_SEARCH_API_KEY={redact(self.parms.get("AZURE_AI_SEARCH_API_KEY"))}, {self.parms.get("AZURE_AI_SEARCH_EMBEDDING_DEPLOYMENT")=}, '
                 f'{self.parms.get("AZURE_AI_SEARCH_EMBEDDING_DIMENSIONS")=}')
        self._aoai_client = openai.AzureOpenAI(azure_endpoint=self.parms.get("AZURE_OPENAI_ENDPOINT"),
                                               api_key=self.parms.get("AZURE_OPENAI_API_KEY"),
                                               api_version=self.parms.get("AZURE_OPENAI_API_VERSION"))

        aai_search_api_key: str = self.parms.get("AZURE_AI_SEARCH_API_KEY")
        if len(aai_search_api_key) > 0:
            search_credential = AzureKeyCredential(aai_search_api_key)
        else:
            search_credential = DefaultAzureCredential()

        self._aais_client = SearchClient(endpoint=self.parms.get("AZURE_AI_SEARCH_ENDPOINT"), index_name=self.index_name,
                                         credential=search_credential)
        self._aais_index_client = SearchIndexClient(endpoint=self.parms.get("AZURE_AI_SEARCH_ENDPOINT"), credential=search_credential)

    def list_index_names(self) -> list[str]:
        self._build_clients()
        return [n for n in self._aais_index_client.list_index_names()]

    def search(self, prompt: str, howmany: int) -> VSAPI.SearchResponse:
        self._build_clients()
        query_embedding = self._aoai_client.embeddings.create(
            input=prompt,
            # todo: curiously this was already specified when the aoai_client was created
            model=self.deployment,
            dimensions=self.embedding_dimensions).data[0].embedding

        vector_query = VectorizedQuery(vector=query_embedding,
                                       k_nearest_neighbors=howmany if howmany > 0 else None,
                                       fields='questionVector',
                                       exhaustive=False)

        # dict str->?: 'source', 'id', '@search.score', '@search.reranker_score', '@search.highlights', '@search.captions'
        # todo: question and answer are alcami-specific
        results: SearchItemPaged[dict] = self._aais_client.search(
            search_text=None,
            vector_queries=[vector_query],
            query_type='semantic',
            semantic_configuration_name='questions-semantic-config',
            select=['question', 'answer', 'source', 'id'],
        )

        raw_results = [d for d in results]
        # todo: answer is alcami-specific
        return VSAPI.SearchResponse(
            results_text=[rr['answer'] for rr in raw_results],
            results_score=[rr['@search.score'] for rr in raw_results],
            results_raw=raw_results
        )

    @staticmethod
    def create(api_type_name: str, index_name: str, parms: dict[str, str]):
        return VSAzure(api_type_name, index_name, parms)
