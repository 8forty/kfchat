import timeit

import dotenv
from dotenv import load_dotenv

from VSAPI import VSAPI

load_dotenv(override=True)

env_values = dotenv.dotenv_values()


def az(api: VSAPI, prompt: str, index_name: str, howmany: int):
    results = api.search(prompt, howmany)

    return results


def run(prompt: str, api_type_name: str, index_name: str):
    start = timeit.default_timer()
    api = VSAPI(api_type_name, env_values)

    print(f'---- generating response from {api.type()}:{index_name}')

    res = az(api, prompt, index_name=index_name, howmany=3)
    rcount = 0
    for result in res:
        print(result)
        rcount += 1

    end = timeit.default_timer()

    print(f'\n{api.type()}:{index_name} '
          f'responded with {rcount} responses '
          f'in {end - start:.0f} seconds\n')


indexes = {
    # 'chroma': ['oregon.pdf'],
    'azure': ['rfibot-qi-index-2024-12-21-00-17-55'],
}
for atype in indexes.keys():
    run('how much lab space?', atype, indexes[atype][0])
