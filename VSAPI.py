import logging

import openai
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents import SearchItemPaged
from azure.search.documents.models import VectorizedQuery

import logstuff
from config import redact

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class VSAPI:
    def __init__(self, api_type: str, parms: dict[str, str]):
        """

        :param api_type: currently: ['azure', 'chroma']
        :param parms: (possibly env vars) that set needed parms for the api, e.g. key, endpoint...
        """
        if api_type in ['azure', 'chroma']:
            self._api_type_name = api_type
            self.parms = parms
            self._api_client_data = None
        else:
            raise ValueError(f'{__class__.__name__}: invalid api_type! {api_type}')

    def __repr__(self) -> str:
        return f'[{self.__class__!s}:{self.__dict__!r}]'

    def type(self) -> str:
        return self._api_type_name

    def _client_data(self):
        if self._api_client_data is not None:
            return self._api_client_data

        if self._api_type_name == "azure":
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
            aoai_client = openai.AzureOpenAI(azure_endpoint=self.parms.get("AZURE_OPENAI_ENDPOINT"),
                                             api_key=self.parms.get("AZURE_OPENAI_API_KEY"),
                                             api_version=self.parms.get("AZURE_OPENAI_API_VERSION"))

            aai_search_api_key: str = self.parms.get("AZURE_AI_SEARCH_API_KEY")
            if len(aai_search_api_key) > 0:
                search_credential = AzureKeyCredential(aai_search_api_key)
            else:
                search_credential = DefaultAzureCredential()

            aais_client = SearchClient(endpoint=self.parms.get("AZURE_AI_SEARCH_ENDPOINT"), index_name='rfibot-qi-index-2024-12-21-00-17-55',
                                       credential=search_credential)

            self._api_client_data = {'aoai': aoai_client, 'aais': aais_client,
                                     'deployment': self.parms.get("AZURE_AI_SEARCH_EMBEDDING_DEPLOYMENT"),
                                     'embedding_dimensions': int(self.parms.get("AZURE_AI_SEARCH_EMBEDDING_DIMENSIONS"))
                                     }


        # elif self._api_type_name == "ollama":
        #     log.info(f'building LLM API for [{self._api_type_name}]: {self.parms.get("OLLAMA_ENDPOINT")=}')
        #     self._api_client = openai.OpenAI(base_url=self.parms.get("OLLAMA_ENDPOINT"),
        #                                      api_key="nokeyneeded")
        # elif self._api_type_name == "openai":
        #     log.info(f'building LLM API for [{self._api_type_name}]: {self.parms.get("OPENAI_ENDPOINT")}, {redact(self.parms.get("OPENAI_API_KEY"))}')
        #     self._api_client = openai.OpenAI(base_url=self.parms.get("OPENAI_ENDPOINT"),
        #                                      api_key=self.parms.get("OPENAI_API_KEY"))
        # elif self._api_type_name == "groq":
        #     log.info(f'building LLM API for [{self._api_type_name}]: {self.parms.get("GROQ_ENDPOINT")}, {redact(self.parms.get("GROQ_API_KEY"))}')
        #     self._api_client = openai.OpenAI(base_url=self.parms.get("GROQ_OPENAI_ENDPOINT"),
        #                                      api_key=self.parms.get("GROQ_API_KEY"))
        # elif self._api_type_name == "github":
        #     base_url = "https://models.inference.ai.azure.com"
        #     log.info(f'building LLM API for [{self._api_type_name}]: {base_url=}, {redact(self.parms.get("GITHUB_TOKEN"))}')
        #     self._api_client = openai.OpenAI(base_url=base_url,
        #                                      api_key=self.parms.get("GITHUB_TOKEN"))
        else:
            raise ValueError(f'invalid api_type! {self._api_type_name}')

        return self._api_client_data

    def search(self, prompt: str, howmany: int):
        aoai_client: openai.AzureOpenAI = self._client_data()['aoai']
        query_embedding = aoai_client.embeddings.create(
            input=prompt,
            # todo: curiously this was already specified when the aoai_client was created
            model=self._client_data()['deployment'],
            dimensions=self._client_data()['embedding_dimensions']).data[0].embedding

        vector_query = VectorizedQuery(vector=query_embedding,
                                       k_nearest_neighbors=howmany if howmany > 0 else None,
                                       fields='questionVector',
                                       exhaustive=False)

        search_client: SearchClient = self._client_data()['aais']
        results: SearchItemPaged[dict] = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            query_type='semantic',
            semantic_configuration_name='questions-semantic-config',
            select=['question', 'answer', 'source', 'id'],
        )

        return results
