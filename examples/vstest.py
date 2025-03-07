import timeit

import dotenv
from dotenv import load_dotenv

from vectorstore import vsapi_factory
from vectorstore.vsapi import VSAPI

load_dotenv(override=True)

env_values = dotenv.dotenv_values()


def run(prompt: str, vs_type_name: str, collection_name: str, howmany: int):
    start = timeit.default_timer()
    vs: VSAPI = vsapi_factory.create_one(vs_type_name, env_values)

    print(f'---- collection names from {vs.type()}:{collection_name}')
    print(f'     {vs.list_collection_names()}')

    vs.switch_collection(collection_name)
    print(f'---- generating response from {vs.type()}:{collection_name}')

    # response = api.search(prompt, howmany)
    # for i in range(0, len(results.results_text)):
    #     print(f'{results.results_score[i]:0.3f}: {results.results_text[i]}')

    end = timeit.default_timer()

    # print(f'\n{api.type()}:{collection_name} '
    #       f'responded with {len(results.results_text)} responses '
    #       f'in {end - start:.0f} seconds\n')


collections = {
    'chroma': ['oregon.pdf'],
    'azure': ['rfibot-qi-index-2024-12-21-00-17-55'],
}
for vtype in collections.keys():
    run('how much lab space?', vtype, collections[vtype][0], 2)
