import timeit

import dotenv
from dotenv import load_dotenv

from vsapi import VSAPI
from vschroma import VSChroma

load_dotenv(override=True)

env_values = dotenv.dotenv_values()


def az(api: VSAPI, prompt: str, index_name: str, howmany: int) -> VSAPI.SearchResponse:
    results = api.search(prompt, howmany)

    return results


def run(prompt: str, api_type_name: str, index_name: str):
    start = timeit.default_timer()
    api = VSChroma(api_type_name, index_name, env_values)

    print(f'---- generating response from {api.type()}:{index_name}')

    results = az(api, prompt, index_name=index_name, howmany=3)
    for i in range(0, len(results.results_text)):
        print(f'{results.results_score[i]:0.3f}: {results.results_text[i]}')

    end = timeit.default_timer()

    print(f'\n{api.type()}:{index_name} '
          f'responded with {len(results.results_text)} responses '
          f'in {end - start:.0f} seconds\n')


indexes = {
    'chroma': ['oregon.pdf'],
    # 'azure': ['rfibot-qi-index-2024-12-21-00-17-55'],
}
for atype in indexes.keys():
    run('how much lab space?', atype, indexes[atype][0])
