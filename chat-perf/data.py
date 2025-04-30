from dataclasses import dataclass
from enum import Enum

import config
from llmconfig.llm_openai_config import LLMOpenAISettings
from llmconfig.llmexchange import LLMMessagePair


class CPRunType(Enum):
    LLM = 1
    VS = 2
    RAG = 3


@dataclass
class CPRunSpec:
    run_type: CPRunType
    model: config.ModelSpec


class Data:
    ##############################################################
    # run sets
    ##############################################################
    run_sets: dict[str, list[CPRunSpec]] = {
        'base': [
            CPRunSpec(CPRunType.LLM, config.LLMData.models_by_pname['OLLAMA.llama3.2:1b']),
            CPRunSpec(CPRunType.LLM, config.LLMData.models_by_pname['OLLAMA.llama3.2:3b']),
        ],

        'groq-base': [
            config.LLMData.models_by_pname['GROQ.llama-3.3-70b-versatile'],
            config.LLMData.models_by_pname['GROQ.qwen-qwq-32b'],
        ],
        'groq-all': [
            config.LLMData.models_by_pname['GROQ.llama-3.3-70b-versatile'],
            config.LLMData.models_by_pname['GROQ.qwen-qwq-32b'],
            config.LLMData.models_by_pname['GROQ.gemma2-9b-it'],
            config.LLMData.models_by_pname['GROQ.deepseek-r1-distill-llama-70b'],
        ],

        'gorbash-test': [
            config.LLMData.models_by_pname['OLLAMA.llama3.2:3b'],
            config.LLMData.models_by_pname['OLLAMA.mistral-nemo:12b'],
            config.LLMData.models_by_pname['OLLAMA.mixtral:8x7b'],
            config.LLMData.models_by_pname['OLLAMA.gemma2:9b-instruct-fp16'],
            config.LLMData.models_by_pname['OLLAMA.gemma2:9b-text-fp16'],
            config.LLMData.models_by_pname['OLLAMA.gemma3:1b'],
            config.LLMData.models_by_pname['OLLAMA.gemma3:4b'],
            config.LLMData.models_by_pname['OLLAMA.gemma3:12b'],
            config.LLMData.models_by_pname['OLLAMA.gemma3:12b-it-fp16'],
            config.LLMData.models_by_pname['OLLAMA.gemma3:27b'],
            config.LLMData.models_by_pname['OLLAMA.gemma3:27b-it-fp16'],
            config.LLMData.models_by_pname['OLLAMA.llama3.3:70b'],
            config.LLMData.models_by_pname['OLLAMA.llama3.3:70b-instruct-q2_K'],
            config.LLMData.models_by_pname['OLLAMA.deepseek-r1:32b'],
            config.LLMData.models_by_pname['OLLAMA.deepseek-v2:16b'],
            config.LLMData.models_by_pname['OLLAMA.qwq:latest'],
            config.LLMData.models_by_pname['OLLAMA.phi4:14b'],
            config.LLMData.models_by_pname['OLLAMA.phi4:14b-q8_0'],
            config.LLMData.models_by_pname['OLLAMA.phi4:14b-fp16'],
            config.LLMData.models_by_pname['OLLAMA.olmo2:13b'],
            config.LLMData.models_by_pname['OLLAMA.command-r7b'],
            config.LLMData.models_by_pname['OLLAMA.openthinker:32b'],
            config.LLMData.models_by_pname['OLLAMA.qwen3:14b-q8_0'],
            config.LLMData.models_by_pname['OLLAMA.qwen3:30b-a3b'],
            config.LLMData.models_by_pname['OLLAMA.qwen3:30b-a3b-q4_K_M'],
            config.LLMData.models_by_pname['OLLAMA.qwen3:32b-q4_K_M'],
            config.LLMData.models_by_pname['OLLAMA.qwen3:32b'],
        ],
    }

    ##############################################################
    # llm settings
    ##############################################################
    class LLMRawSettings(LLMOpenAISettings):
        def __init__(self, init_n: int, init_temp: float, init_top_p: float, init_max_tokens: int,
                     init_system_message_name: str):
            """
            (almost) standard set of settings for LLMs
            todo: some LLMs/providers don't support n
            :param init_n:
            :param init_temp:
            :param init_top_p:
            :param init_max_tokens:
            :param init_system_message_name:

            """
            super().__init__(init_n=init_n, init_temp=init_temp, init_top_p=init_top_p,
                             init_max_tokens=init_max_tokens, init_system_message_name=init_system_message_name)

    #         'convo': conversational_sysmsg,
    #         'convo80': conversational80_sysmsg,
    #         'professional': professional_sysmsg,
    #         'professional80': professional80_sysmsg,
    #         'professional800': professional800_sysmsg,
    #         'technical': technical_sysmsg,
    #         'technical80': technical80_sysmsg,
    #         'technical800': technical800_sysmsg,
    #         'text-sentiment': textclass_sysmsg,
    #         'carl-sagan': csagan_sysmsg,
    #         'empty': empty_sysmsg,
    llm_settings_sets = {
        '1:800': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=800,
                           init_system_message_name='empty'),
        ],
        'quick': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=80,
                           init_system_message_name='empty'),
        ],
        'std4': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=800,
                           init_system_message_name='empty'),
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=400,
                           init_system_message_name='empty'),
            LLMRawSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=800,
                           init_system_message_name='empty'),
            LLMRawSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=400,
                           init_system_message_name='empty'),
        ],
        'gorbash-test': [
            LLMRawSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=800,
                           init_system_message_name='professional800'),
        ],
        'ollama-warmup': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=800,
                           init_system_message_name='carl-sagan'),
        ]
    }

    ##############################################################
    # prompt sets
    ##############################################################

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
    llm_prompt_sets = {
        'space': [
            [LLMMessagePair('user', galaxies_prompt)],
            [LLMMessagePair('user', blackholes_prompt)]
        ],
        'explain': [
            [LLMMessagePair('user', explain_prompt)]
        ],
        'onesentence': [
            [LLMMessagePair('user', onesentence_prompt)]
        ],
        'info': [
            [LLMMessagePair('user', info_prompt)]
        ],
        'drug': [
            [LLMMessagePair('user', teplizumab_prompt)]
        ],
        'gorbash-test': [
            # 'what data security does gorbash have?'
            [LLMMessagePair('user', 'gorbash compliance hotline number?')]
        ],
    }
