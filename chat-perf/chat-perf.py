import sys
import timeit

import openai

import config
import data
from llmapi import LLMAPI, LLMExchange


def model_warmup(api: LLMAPI, model: str):
    chat(message_set=data.warmup_data['messageset'],
         api=api,
         model_name=model,
         temp=data.warmup_data['temp'],
         max_tokens=data.warmup_data['max_tokens'])


def chat(message_set: list[tuple[str, str]], api: LLMAPI, model_name: str, temp: float, max_tokens: int) -> LLMExchange:
    messages: list[dict] = []
    for i in message_set:
        messages.append({'role': i[0], 'content': i[1]})

    try:
        return api.run_chat_completion(
            model_name=model_name,
            temp=temp,  # default 1.0, 0.0->2.0
            top_p=1.0,  # default 1, ~0.01->1.0
            max_tokens=max_tokens,  # default 16?
            n=1,
            convo=messages,

            # stream=False,
            # seed=27,
            # frequency_penalty=1,  # default 0, -2.0->2.0
            # presence_penalty=1,  # default 0, -2.0->2.0
            # stop=[],
        )
    except (Exception,) as e:
        print(f'chat Exception! {model_name}:  {e}')
        raise


def run(api_type_name: str, model_set_name: str, settings_set_name: str, message_sets_name: str):
    csv_data = []

    # API
    print(f'---- running {api_type_name} {model_set_name} {settings_set_name} {message_sets_name}...')
    api_start_time = timeit.default_timer()

    # model
    all_start = timeit.default_timer()
    for model in data.model_sets[api_type_name][model_set_name]:
        print(f'    {api_type_name}:{model}')
        model_start = timeit.default_timer()
        api = LLMAPI(api_type_name, parms=data.model_sets[api_type_name]['parms'])

        # warmup the model
        try:
            model_warmup(api, model)
        except (Exception,) as e:
            print(f'warmup Exception! {api_type_name}:{model}: {e} skipping...')
            break

        print(f'{config.secs_string(all_start)}: [{api.type()}:{model}] model-warmup: [{timeit.default_timer() - model_start:.0f}]s')
        csv_data.append([api.type(), model, '', '', '(warm-up)', '', '', str(int(timeit.default_timer() - model_start)), ''])

        # settings
        for settings_set in data.settings_sets[settings_set_name]:

            # message sets
            for message_set in data.message_sets[message_sets_name]:
                ms_start = timeit.default_timer()
                try:
                    exchange = chat(message_set=message_set['messages'],
                                    api=api,
                                    model_name=model,
                                    temp=settings_set['temp'],
                                    max_tokens=settings_set['max_tokens'])
                except (Exception,) as e:
                    print(f'run Exception! {api.type()}:{model} {settings_set_name} {message_set["name"]}: {e} skipping...')
                    break

                ms_end = timeit.default_timer()

                print(f'{config.secs_string(all_start)}: [{api.type()}:{model}] [{settings_set['temp']}] [{settings_set['max_tokens']}] [{message_set["name"]}]: '
                      f'[{exchange.completion.usage.prompt_tokens}+{exchange.completion.usage.completion_tokens}] '
                      f'[{ms_end - ms_start:.0f}]s')
                response_1line = str(exchange.completion.choices[0].message.content).replace("\n", "  ").replace('"', '""')
                print(f'{config.secs_string(all_start)}:     {response_1line}')
                csv_data.append([api.type(), model, str(settings_set['temp']), str(settings_set['max_tokens']),
                                 message_set["name"],
                                 str(exchange.completion.usage.prompt_tokens), str(exchange.completion.usage.completion_tokens),
                                 str(int(ms_end - ms_start)),
                                 '"' + response_1line + '"'],
                                )

        model_end = timeit.default_timer()
        print(f'{config.secs_string(all_start)}: [{api.type()}:{model}] [{model_end - model_start:.0f}]s')
        csv_data.append([api.type(), model, '', '', '', '', '', str(int(model_end - model_start)), ''])

    api_end_time = timeit.default_timer()
    print(f'{config.secs_string(all_start)}: [{api_type_name}] [{api_end_time - api_start_time:.0f}]s')

    # now make CSV lines from results
    csv_data.append([api_type_name, '', '', '', '', '', '', str(int(api_end_time - api_start_time)), ''])

    print('\n\n')
    for line in csv_data:
        print(','.join(line))


run(api_type_name='ollama', model_set_name='mistral7b', settings_set_name='1:800', message_sets_name='space')
# run(api='groq', model_set_name='ll33', settings_set_name='1:800', message_sets_name='text')

# run(api='ollama', model_set_name='mistral7b', settings_set_name='std4', message_sets_name='std7')

# run(api='ollama', model_set_name='ll33', settings_set_name='1:800', message_sets_name='std7')
# run(api='groq', model_set_name='ll33', settings_set_name='1:800', message_sets_name='std7')

# run(api='ollama', model_set_name='std8', settings_set_name='1:800', message_sets_name='text')
# run(api='ollama', model_set_name='std8', settings_set_name='std4', message_sets_name='std7')

# run(api='ollama', model_set_name='ll1b', settings_set_name='1:800', message_sets_name='carl')

# run(api='openai', model_set_name='std3', settings_set_name='1:800', message_sets_name='carl')
# run(api='openai', model_set_name='4omini', settings_set_name='std4', message_sets_name='std7')

# run(api='azure', model_set_name='4omini', settings_set_name='1:800', message_sets_name='carl')
# run(api='azure', model_set_name='4omini', settings_set_name='std4', message_sets_name='std7')
