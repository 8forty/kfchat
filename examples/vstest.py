import os
import timeit

import dotenv
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents import SearchItemPaged
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from openai import AzureOpenAI

from VSAPI import VSAPI

load_dotenv(override=True)

env_values = dotenv.dotenv_values()

aoai_endpoint: str = os.getenv('AZURE_OPENAI_ENDPOINT')
aoai_api_key: str = os.getenv('AZURE_OPENAI_API_KEY')
aoai_api_version: str = os.getenv('AZURE_OPENAI_API_VERSION')

aai_search_endpoint: str = os.getenv('AZURE_AI_SEARCH_ENDPOINT')
aai_search_api_key: str = os.getenv('AZURE_AI_SEARCH_API_KEY')
deployment = 'RFI-Automate-text-embedding-3-large'
embedding_dimensions = 1024

aoai_client: AzureOpenAI = AzureOpenAI(azure_endpoint=aoai_endpoint, api_key=aoai_api_key,
                                       api_version=aoai_api_version, azure_deployment=deployment)

if len(aai_search_api_key) > 0:
    search_credential = AzureKeyCredential(aai_search_api_key)
else:
    search_credential = DefaultAzureCredential()
search_index_client = SearchIndexClient(endpoint=aai_search_endpoint, credential=search_credential)


def az(prompt: str, index_name: str, howmany: int):
    search_client = SearchClient(endpoint=aai_search_endpoint, index_name=index_name,
                                 credential=search_credential)

    query_embedding = aoai_client.embeddings.create(
        input=prompt,
        model=deployment,  # todo: curiously this was already specified when the aoai_client was created
        dimensions=embedding_dimensions).data[0].embedding

    vector_query = VectorizedQuery(vector=query_embedding,
                                   k_nearest_neighbors=howmany if howmany > 0 else None,
                                   fields='questionVector',
                                   exhaustive=False)

    results: SearchItemPaged[dict] = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        query_type='semantic',
        semantic_configuration_name='questions-semantic-config',
        select=['question', 'answer', 'source', 'id'],
    )

    return results


def run(prompt: str, api_type_name: str, index_name: str):
    start = timeit.default_timer()
    api = VSAPI(api_type_name, env_values)

    print(f'---- generating response from {api.type()}:{index_name}')

    res = az(prompt, index_name=index_name, howmany=3)
    rcount = 0
    for result in res:
        print(result)
        rcount += 1

    end = timeit.default_timer()

    print(f'\n{api.type()}:{index_name} '
          f'responded with {rcount} responses '
          f'in {end - start:.0f} seconds')


indexes = {
    'chroma': ['oregon.pdf'],
    'azure': ['rfibot-qi-index-2024-12-21-00-17-55'],
}
for atype in indexes.keys():
    run('how much lab space?', atype, indexes[atype][0])
    print('\n')
