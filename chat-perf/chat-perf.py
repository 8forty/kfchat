import sys
import timeit
import traceback

import config
from llmconfig.llm_anthropic_config import LLMAnthropicConfig
from llmconfig.llm_openai_config import LLMOpenAIConfig, LLMOpenAISettings
from llmconfig.llmconfig import LLMConfig
from llmconfig.llmexchange import LLMMessagePair
from llmconfig.llmsettings import LLMSettings


class Data:
    llm_model_sets = {
        'base': [config.LLMData.models_by_pname['ollama.llama3.2:1b']],

        'grog-base': [
            config.LLMData.models_by_pname['groq.llama-3.3-70b-versatile'],
            config.LLMData.models_by_pname['groq.mixtral-8x7b-32768'],
        ],
        'grog-all': [
            config.LLMData.models_by_pname['groq.llama-3.3-70b-versatile'],
            config.LLMData.models_by_pname['groq.mixtral-8x7b-32768'],
            config.LLMData.models_by_pname['groq.gemma2-9b-it'],
            config.LLMData.models_by_pname['groq.deepseek-r1-distill-llama-70b'],
        ],
    }

    class LLMRawSettings(LLMSettings):
        def __init__(self, init_n: int, init_temp: float, init_top_p: float, init_max_tokens: int, init_system_message_name: str):
            """
            (almost) standard set of settings for LLMs
            todo: some LLMs/providers don't support n
            :param init_n:
            :param init_temp:
            :param init_top_p:
            :param init_max_tokens:
            :param init_system_message_name:

            """
            super().__init__(init_n=init_n, init_temp=init_temp, init_top_p=init_top_p, init_max_tokens=init_max_tokens, init_system_message_name=init_system_message_name)

    llm_settings_sets = {
        '1:800': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=800, init_system_message_name='empty'),
        ],
        'quick': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=80, init_system_message_name='empty'),
        ],
        'std4': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=800, init_system_message_name='empty'),
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=400, init_system_message_name='empty'),
            LLMRawSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=800, init_system_message_name='empty'),
            LLMRawSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=400, init_system_message_name='empty'),
        ],
        'ollama-warmup': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=800, init_system_message_name='carl-sagan'),
        ]
    }

    galaxies_prompt = 'How many galaxies are there?'
    explain_prompt = 'Explain antibiotics'
    onesentence_prompt = ('Antibiotics are a type of medication used to treat bacterial infections. They work by either killing '
                          'the bacteria or preventing them from reproducing, allowing the body’s immune system to fight off the infection. '
                          'Antibiotics are usually taken orally in the form of pills, capsules, or liquid solutions, or sometimes '
                          'administered intravenously. They are not effective against viral infections, and using them '
                          'inappropriately can lead to antibiotic resistance. Explain the above in one sentence:')
    info_prompt = ('Author-contribution statements and acknowledgements in research papers should state clearly and specifically '
                   'whether, and to what extent, the authors used AI technologies such as ChatGPT in the preparation of their '
                   'manuscript and analysis. They should also indicate which LLMs were used. This will alert editors and reviewers '
                   'to scrutinize manuscripts more carefully for potential biases, inaccuracies and improper source crediting. '
                   'Likewise, scientific journals should be transparent about their use of LLMs, for example when selecting '
                   'submitted manuscripts.  Mention the large language model based product mentioned in the paragraph above:')
    teplizumab_prompt = ('Context: Teplizumab traces its roots to a New Jersey drug company called Ortho Pharmaceutical. There, '
                         'scientists generated an early version of the antibody, dubbed OKT3. Originally sourced from mice, '
                         'the molecule was able to bind to the surface of T cells and limit their cell-killing potential. '
                         'In 1986, it was approved to help prevent organ rejection after kidney transplants, making it the '
                         'first therapeutic antibody allowed for human use.  \nQuestion: What was OKT3 originally sourced from?')
    neutralfood_prompt = 'I think the food was okay.'
    blackholes_prompt = 'Can you tell me about the creation of blackholes?'
    rag_lc_rlm_prompt = ("You are an assistant for question-answering tasks. Use the following pieces of retrieved context"
                         " to answer the question. If you don't know the answer, just say that you don't know. Use three "
                         "sentences maximum and keep the answer concise.  \nQuestion: {question}  \nContext: {context}  \nAnswer:")

    # each value is a list of message-sets (i.e. lists of LLMMessagePair's] to run
    llm_message_sets = {
        'space': [[LLMMessagePair('user', galaxies_prompt)], [LLMMessagePair('user', blackholes_prompt)]],
        'explain': [[LLMMessagePair('user', explain_prompt)]],
        'onesentence': [[LLMMessagePair('user', onesentence_prompt)]],
        'info': [[LLMMessagePair('user', info_prompt)]],
        'drug': [[LLMMessagePair('user', teplizumab_prompt)]],
    }


def run(model_sets_name: str, settings_sets_name: str, message_sets_name: str):
    csv_data = []

    run_start_time = timeit.default_timer()
    model_spec: config.LLMData.ModelSpec

    # llm_model_sets
    for model_spec in Data.llm_model_sets[model_sets_name]:
        settings: Data.LLMRawSettings
        print(f'{config.secs_string(run_start_time)}: running {model_spec.provider} {model_spec.name}...')

        # warmup the model if necessary
        if model_spec.provider == 'ollama':
            warmup_start = timeit.default_timer()
            try:
                print(f'{config.secs_string(run_start_time)}: warmup {model_spec.provider} {model_spec.name}...')
                if model_spec.api == 'openai':
                    llm_config = LLMOpenAIConfig(model_spec.name, model_spec.provider, LLMOpenAISettings.from_settings(Data.llm_settings_sets['ollama-warmup'][0]))
                elif model_spec.api == 'anthropic':
                    llm_config = LLMAnthropicConfig(model_spec.name, model_spec.provider, LLMOpenAISettings.from_settings(Data.llm_settings_sets['ollama-warmup'][0]))
                else:
                    raise ValueError(f'api must be "openai" or "anthropic"!')
                llm_config.do_chat([LLMMessagePair('user', 'How many galaxies are there?')])
                warmup_secs = timeit.default_timer() - warmup_start
                csv_data.append([llm_config.provider(), llm_config.model_name, '', '', '(warm-up)', '', '', str(int(warmup_secs))])
                print(f'{config.secs_string(run_start_time)}: warmup: {warmup_secs:.1f}s')
            except (Exception,) as e:
                print(f'{config.secs_string(run_start_time)}: warmup Exception! {model_spec.provider}:{model_spec.name}: {e.__class__.__name__}: {e} skipping...')
                traceback.print_exc(file=sys.stderr)
                break

        model_start = timeit.default_timer()

        # llm_settings_sets
        for settings in Data.llm_settings_sets[settings_sets_name]:
            if model_spec.api == 'openai':
                llm_config = LLMOpenAIConfig(model_spec.name, model_spec.provider, LLMOpenAISettings.from_settings(settings))
            elif model_spec.api == 'anthropic':
                llm_config = LLMAnthropicConfig(model_spec.name, model_spec.provider, LLMOpenAISettings.from_settings(settings))
            else:
                raise ValueError(f'api must be "openai" or "anthropic"!')

            # llm_message_sets
            ms_start = timeit.default_timer()
            ms_input_tokens = 0
            ms_output_tokens = 0
            for idx, message_set in enumerate(Data.llm_message_sets[message_sets_name]):
                try:
                    exchange = llm_config.do_chat(message_set)
                except (Exception,) as e:
                    print(f'run Exception! {llm_config.provider()}:{llm_config.model_name} {message_sets_name}.{message_set}: {e.__class__.__name__}: {e} skipping...')
                    break

                ms_input_tokens += exchange.input_tokens
                ms_output_tokens += exchange.output_tokens
                response_line = str(exchange.responses[0].content).replace("\n", "  ").replace('"', '""')
                print(f'{config.secs_string(run_start_time)}: {message_sets_name}[{idx}]: '
                      f'{exchange.input_tokens}+{exchange.output_tokens} '
                      f'{timeit.default_timer() - ms_start:.1f}s  {response_line}')

            ms_end = timeit.default_timer()
            print(f'{config.secs_string(run_start_time)}: {message_sets_name}: [{llm_config.provider()}:{llm_config.model_name}] {llm_config.settings().temp}/{llm_config.settings().max_tokens}: '
                  f'{ms_input_tokens}+{ms_output_tokens} '
                  f'{timeit.default_timer() - ms_start:.1f}s')
            csv_data.append([llm_config.provider(), llm_config.model_name, str(llm_config.settings().temp), str(llm_config.settings().max_tokens),
                             f'{message_sets_name}',
                             str(ms_input_tokens), str(ms_output_tokens),
                             str(ms_end - ms_start)]
                            )

        model_end = timeit.default_timer()
        print(f'{config.secs_string(run_start_time)}: [{model_spec.provider}:{model_spec.name}] {model_end - model_start:.1f}s')
        csv_data.append([model_spec.provider, model_spec.name, '', '', '', '', '', str(model_end - model_start)])

    run_end_time = timeit.default_timer()
    print(f'{run_end_time - run_start_time:.1f}s')

    print('\n\n')
    # now make CSV lines from results
    for line in csv_data:
        print(','.join(line))


run(model_sets_name='base', settings_sets_name='quick', message_sets_name='space')
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
