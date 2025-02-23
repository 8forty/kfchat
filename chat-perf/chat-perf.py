import timeit

import config
import data
from llmconfig.llm_anthropic_config import LLMAnthropicSettings
from llmconfig.llmconfig import LLMConfig
from llmconfig.llm_openai_config import LLMOpenAISettings, LLMOpenAIConfig


def model_warmup(cfg: LLMConfig, model: str):
    try:
        cfg.chat_messages(data.warmup_data['messageset'])
    except (Exception,) as e:
        print(f'chat Exception! {model}: {e.__class__.__name__}: {e}')
        raise


def run(cfgs: [LLMConfig], message_sets_name: str):
    csv_data = []

    api_start_time = timeit.default_timer()
    all_start = timeit.default_timer()
    llm_config: LLMConfig
    for llm_config in cfgs:

        # cfg
        print(f'---- running {llm_config.provider()} {llm_config.model_name} {llm_config.settings()} {message_sets_name}...')
        model_start = timeit.default_timer()

        # warmup the model
        try:
            model_warmup(llm_config, llm_config.model_name)
        except (Exception,) as e:
            print(f'warmup Exception! {llm_config.provider()}:{llm_config.model_name}: {e.__class__.__name__}: {e} skipping...')
            break

        print(f'{config.secs_string(all_start)}: [{llm_config.provider()}:{llm_config.model_name}] model-warmup: [{timeit.default_timer() - model_start:.0f}]s')
        csv_data.append([llm_config.provider(), llm_config.model_name, '', '', '(warm-up)', '', '', str(int(timeit.default_timer() - model_start)), ''])

        # message sets
        for message_set in data.message_sets[message_sets_name]:
            ms_start = timeit.default_timer()
            try:
                exchange = llm_config.chat_messages(message_set['messages'])
            except (Exception,) as e:
                print(f'run Exception! {llm_config.provider()}:{llm_config.model_name} {message_set["name"]}: {e.__class__.__name__}: {e} skipping...')
                break

            ms_end = timeit.default_timer()

            print(f'{config.secs_string(all_start)}: [{llm_config.provider()}:{llm_config.model_name}] [{llm_config.settings().temp}] [{llm_config.settings().max_tokens}]: '
                  f'[{exchange.input_tokens}+{exchange.output_tokens}] '
                  f'[{ms_end - ms_start:.0f}]s')
            response_line = str(exchange.responses[0].content).replace("\n", "  ").replace('"', '""')
            print(f'{config.secs_string(all_start)}:     {response_line}')
            csv_data.append([llm_config.provider(), llm_config.model_name, str(llm_config.settings().temp), str(llm_config.settings().max_tokens),
                             message_set["name"],
                             str(exchange.input_tokens), str(exchange.output_tokens),
                             str(int(ms_end - ms_start)),
                             '"' + response_line + '"'],
                            )

        model_end = timeit.default_timer()
        print(f'{config.secs_string(all_start)}: [{llm_config.provider()}:{llm_config.model_name}] [{model_end - model_start:.0f}]s')
        csv_data.append([llm_config.provider(), llm_config.model_name, '', '', '', '', '', str(int(model_end - model_start)), ''])

    api_end_time = timeit.default_timer()
    print(f'{config.secs_string(all_start)}: [{api_end_time - api_start_time:.0f}]s')

    # now make CSV lines from results
    csv_data.append(['bad-provider', '', '', '', '', '', '', str(int(api_end_time - api_start_time)), ''])

    print('\n\n')
    for line in csv_data:
        print(','.join(line))


settings_openai = LLMOpenAISettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=80, init_system_message_name='carl-sagan')
settings_anthropic = LLMAnthropicSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=80, init_system_message_name='carl-sagan')

run([LLMOpenAIConfig(model_name='llama3.2:1b', provider='ollama', settings=settings_openai)], message_sets_name='space')
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
