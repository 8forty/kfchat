import sys
import timeit

import openai

import config
import data
from llmapi import LLMAPI


def model_warmup(client: openai.OpenAI, model: str):
    chat(message_set=data.warmup_data['messageset'],
         client=client,
         model_name=model,
         temp=data.warmup_data['temp'],
         max_tokens=data.warmup_data['max_tokens'])


def chat(message_set: list[tuple[str, str]], client: openai.OpenAI, model_name: str, temp: float, max_tokens: int) -> openai.ChatCompletion:
    messages: list[dict[str, str]] = []
    for i in message_set:
        messages.append({'role': i[0], 'content': i[1]})
    try:
        return client.chat.completions.create(
            model=model_name,
            temperature=temp,  # default 1.0, 0.0->2.0
            messages=messages,
            max_tokens=max_tokens,  # default 16?

            stream=False,

            # seed=27,
            # n=1,
            # top_p=1,  # default 1, ~0.01->1.0
            # frequency_penalty=1,  # default 0, -2.0->2.0
            # presence_penalty=1,  # default 0, -2.0->2.0
            # stop=[],

        )
    except (Exception,) as e:
        print(f'chat Exception! {model_name}:  {e}')
        raise


def run(api: str, model_set_name: str, settings_set_name: str, message_sets_name: str):
    csv_data = []

    # API
    print(f'---- running {api} {model_set_name} {settings_set_name} {message_sets_name}...')
    api_start = timeit.default_timer()

    # model
    all_start = timeit.default_timer()
    for model in data.model_sets[api][model_set_name]:
        print(f'    {api}:{model}')
        model_start = timeit.default_timer()
        api_type = LLMAPI(api, parms=data.model_sets[api]['parms'])
        client = api_type.client()

        # warmup the model
        try:
            model_warmup(client, model)
        except (Exception,) as e:
            print(f'warmup Exception! {api}:{model}: {e} skipping...')
            break

        print(f'{config.secs_string(all_start)}: [{api_type.type()}:{model}] model-warmup: [{timeit.default_timer() - model_start:.0f}]s')
        csv_data.append([api_type.type(), model, '', '', '(warm-up)', '', '', str(int(timeit.default_timer() - model_start)), ''])

        # settings
        for settings_set in data.settings_sets[settings_set_name]:

            # message sets
            for message_set in data.message_sets[message_sets_name]:
                ms_start = timeit.default_timer()
                try:
                    response = chat(message_set=message_set['messages'],
                                    client=client,
                                    model_name=model,
                                    temp=settings_set['temp'],
                                    max_tokens=settings_set['max_tokens'])
                except (Exception,) as e:
                    print(f'run Exception! {api}:{model} {settings_set_name} {message_set["name"]}: {e} skipping...')
                    break

                ms_end = timeit.default_timer()

                if response is not None:
                    print(f'{config.secs_string(all_start)}: [{api_type.type()}:{model}] [{settings_set['temp']}] [{settings_set['max_tokens']}] [{message_set["name"]}]: '
                          f'[{response.usage.prompt_tokens}+{response.usage.completion_tokens}] '
                          f'[{ms_end - ms_start:.0f}]s')
                    response_1line = str(response.choices[0].message.content).replace("\n", "  ").replace('"', '""')
                    print(f'{config.secs_string(all_start)}:     {response_1line}')
                    csv_data.append([api_type.type(), model, str(settings_set['temp']), str(settings_set['max_tokens']),
                                     message_set["name"],
                                     str(response.usage.prompt_tokens), str(response.usage.completion_tokens),
                                     str(int(ms_end - ms_start)),
                                     '"' + response_1line + '"'],
                                    )

        model_end = timeit.default_timer()
        print(f'{config.secs_string(all_start)}: [{api_type.type()}:{model}] [{model_end - model_start:.0f}]s')
        csv_data.append([api_type.type(), model, '', '', '', '', '', str(int(model_end - model_start)), ''])

    api_end = timeit.default_timer()
    print(f'{config.secs_string(all_start)}: [{api}] [{api_end - api_start:.0f}]s')
    csv_data.append([api, '', '', '', '', '', '', str(int(api_end - api_start)), ''])

    print('\n\n')
    for line in csv_data:
        print(','.join(line))


# run(api='ollama', model_set_name='mstl7b', settings_set_name='1:800', message_sets_name='text')
# run(api='groq', model_set_name='ll33', settings_set_name='1:800', message_sets_name='text')

run(api='ollama', model_set_name='mistral7b', settings_set_name='std4', message_sets_name='std7')

# run(api='ollama', model_set_name='ll33', settings_set_name='1:800', message_sets_name='std7')
# run(api='groq', model_set_name='ll33', settings_set_name='1:800', message_sets_name='std7')

# run(api='ollama', model_set_name='std8', settings_set_name='1:800', message_sets_name='text')
# run(api='ollama', model_set_name='std8', settings_set_name='std4', message_sets_name='std7')

# run(api='ollama', model_set_name='ll1b', settings_set_name='1:800', message_sets_name='carl')

# run(api='openai', model_set_name='std3', settings_set_name='1:800', message_sets_name='carl')
# run(api='openai', model_set_name='4omini', settings_set_name='std4', message_sets_name='std7')

# run(api='azure', model_set_name='4omini', settings_set_name='1:800', message_sets_name='carl')
# run(api='azure', model_set_name='4omini', settings_set_name='std4', message_sets_name='std7')
