import timeit

import config
import data
from llmconfig.llmoaiconfig import LLMOaiConfig, LLMOaiExchange, LLMOaiSettings


def model_warmup(cfg: LLMOaiConfig, model: str):
    chat(message_set=data.warmup_data['messageset'],
         cfg=cfg,
         model_name=model)


def chat(message_set: list[tuple[str, str]], cfg: LLMOaiConfig, model_name: str) -> LLMOaiExchange:
    messages: list[dict] = []
    for i in message_set:
        messages.append({'role': i[0], 'content': i[1]})

    try:
        return cfg.chat_messages(messages)
    except (Exception,) as e:
        print(f'chat Exception! {model_name}: {e.__class__.__name__}: {e}')
        raise


def run(provider_name: str, model_set_name: str, settings_set_name: str, message_sets_name: str):
    csv_data = []

    # API
    print(f'---- running {provider_name} {model_set_name} {settings_set_name} {message_sets_name}...')
    api_start_time = timeit.default_timer()

    # model
    all_start = timeit.default_timer()
    for model in data.model_sets[provider_name][model_set_name]:
        print(f'    {provider_name}:{model}')
        model_start = timeit.default_timer()
        cfg = LLMOaiConfig(model, provider_name,
                           LLMOaiSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=80, init_system_message_name="carl-sagan"))

        # warmup the model
        try:
            model_warmup(cfg, model)
        except (Exception,) as e:
            print(f'warmup Exception! {provider_name}:{model}: {e.__class__.__name__}: {e} skipping...')
            break

        print(f'{config.secs_string(all_start)}: [{cfg.provider()}:{model}] model-warmup: [{timeit.default_timer() - model_start:.0f}]s')
        csv_data.append([cfg.provider(), model, '', '', '(warm-up)', '', '', str(int(timeit.default_timer() - model_start)), ''])

        # settings
        for settings_set in data.settings_sets[settings_set_name]:

            # message sets
            for message_set in data.message_sets[message_sets_name]:
                ms_start = timeit.default_timer()
                try:
                    exchange = chat(message_set=message_set['messages'],
                                    cfg=cfg,
                                    model_name=model)
                except (Exception,) as e:
                    print(f'run Exception! {cfg.provider()}:{model} {settings_set_name} {message_set["name"]}: {e.__class__.__name__}: {e} skipping...')
                    break

                ms_end = timeit.default_timer()

                print(f'{config.secs_string(all_start)}: [{cfg.provider()}:{model}] [{settings_set['temp']}] [{settings_set['max_tokens']}] [{message_set["name"]}]: '
                      f'[{exchange.completion.usage.prompt_tokens}+{exchange.completion.usage.completion_tokens}] '
                      f'[{ms_end - ms_start:.0f}]s')
                response_1line = str(exchange.completion.choices[0].message.content).replace("\n", "  ").replace('"', '""')
                print(f'{config.secs_string(all_start)}:     {response_1line}')
                csv_data.append([cfg.provider(), model, str(settings_set['temp']), str(settings_set['max_tokens']),
                                 message_set["name"],
                                 str(exchange.completion.usage.prompt_tokens), str(exchange.completion.usage.completion_tokens),
                                 str(int(ms_end - ms_start)),
                                 '"' + response_1line + '"'],
                                )

        model_end = timeit.default_timer()
        print(f'{config.secs_string(all_start)}: [{cfg.provider()}:{model}] [{model_end - model_start:.0f}]s')
        csv_data.append([cfg.provider(), model, '', '', '', '', '', str(int(model_end - model_start)), ''])

    api_end_time = timeit.default_timer()
    print(f'{config.secs_string(all_start)}: [{provider_name}] [{api_end_time - api_start_time:.0f}]s')

    # now make CSV lines from results
    csv_data.append([provider_name, '', '', '', '', '', '', str(int(api_end_time - api_start_time)), ''])

    print('\n\n')
    for line in csv_data:
        print(','.join(line))


run(provider_name='ollama', model_set_name='ll1b', settings_set_name='1:800', message_sets_name='space')
# run(provider_name='ollama', model_set_name='mistral7b', settings_set_name='1:800', message_sets_name='space')
# run(provider_name='groq', model_set_name='ll33', settings_set_name='1:800', message_sets_name='text')

# run(provider_name='ollama', model_set_name='mistral7b', settings_set_name='std4', message_sets_name='std7')

# run(provider_name='ollama', model_set_name='ll33', settings_set_name='1:800', message_sets_name='std7')
# run(provider_name='groq', model_set_name='ll33', settings_set_name='1:800', message_sets_name='std7')

# run(provider_name='ollama', model_set_name='std8', settings_set_name='1:800', message_sets_name='text')
# run(provider_name='ollama', model_set_name='std8', settings_set_name='std4', message_sets_name='std7')

# run(provider_name='ollama', model_set_name='ll1b', settings_set_name='1:800', message_sets_name='carl')

# run(provider_name='openai', model_set_name='std3', settings_set_name='1:800', message_sets_name='carl')
# run(provider_name='openai', model_set_name='4omini', settings_set_name='std4', message_sets_name='std7')

# run(provider_name='azure', model_set_name='4omini', settings_set_name='1:800', message_sets_name='carl')
# run(provider_name='azure', model_set_name='4omini', settings_set_name='std4', message_sets_name='std7')
