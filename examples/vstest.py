import timeit

import dotenv
from dotenv import load_dotenv

import vsapi_factory
from vsapi import VSAPI

load_dotenv(override=True)

env_values = dotenv.dotenv_values()


def run(prompt: str, api_type_name: str, index_name: str, howmany: int):
    start = timeit.default_timer()
    api: VSAPI = vsapi_factory.create_one(api_type_name, env_values)

    print(f'---- index names from {api.type()}:{index_name}')
    print(f'     {api.list_index_names()}')

    api.change_index(index_name)
    print(f'---- generating response from {api.type()}:{index_name}')

    results = api.search(prompt, howmany)
    for i in range(0, len(results.results_text)):
        print(f'{results.results_score[i]:0.3f}: {results.results_text[i]}')

    end = timeit.default_timer()

    print(f'\n{api.type()}:{index_name} '
          f'responded with {len(results.results_text)} responses '
          f'in {end - start:.0f} seconds\n')


indexes = {
    'chroma': ['oregon.pdf'],
    'azure': ['rfibot-qi-index-2024-12-21-00-17-55'],
}
for atype in indexes.keys():
    run('how much lab space?', atype, indexes[atype][0], 2)
