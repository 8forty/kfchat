import requests


# {'models': [{'name': 'llama3.2:1b', 'model': 'llama3.2:1b', 'size': 2712725504, 'digest': 'baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878', 'details': {'parent_model': '', 'format': 'gguf', 'family': 'llama', 'families': ['llama'], 'parameter_size': '1.2B', 'quantization_level': 'Q8_0'}, 'expires_at': '2025-05-01T10:53:51.3747775-07:00', 'size_vram': 2712725504}]}
def ps() -> dict:
    response = requests.get('http://localhost:11434/api/ps')
    return response.json()


def unload_all() -> list[dict]:
    models = [m['name'] for m in ps()['models']]
    responses = []
    for model in models:
        url = 'http://localhost:11434/api/generate'
        responses.append(requests.post(url,
                                       headers={'Content-Type': 'application/x-www-form-urlencoded', },
                                       data=f'{{  "model": "{model}",  "keep_alive": 0}}'))
    return responses


print(f'currently loaded: {[m['name'] for m in ps()['models']]}')
print(f'unload responses: {unload_all()}')
