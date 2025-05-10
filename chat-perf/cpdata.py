from dataclasses import dataclass
from enum import Enum

import config
from basesettings import BaseSettings
from llmconfig.llmexchange import LLMMessagePair
from llmconfig.llmsettings import LLMSettings


class CPRunType(Enum):
    LLM = 1
    VS = 2
    RAG = 3


@dataclass
class CPRunSpec:
    run_type: CPRunType
    model: config.ModelSpec
    collection_name: str = ''
    ollama_ctx_size: int = 2048
    seed: int = 0


class CPData:
    ##############################################################
    # run sets
    ##############################################################
    run_sets: dict[str, list[CPRunSpec]] = {
        'kf': [
            CPRunSpec(CPRunType.LLM, config.LLMData.models_by_pname['OLLAMA.gemma3:1b']),
            CPRunSpec(CPRunType.LLM, config.LLMData.models_by_pname['OLLAMA.llama3.2:3b']),
        ],

        'base': [
            CPRunSpec(CPRunType.LLM, config.LLMData.models_by_pname['OLLAMA.llama3.2:1b']),
            # CPRunSpec(CPRunType.LLM, config.LLMData.models_by_pname['OLLAMA.llama3.2:3b']),
        ],

        'groq-base': [
            CPRunSpec(CPRunType.LLM, config.LLMData.models_by_pname['GROQ.qwen-qwq-32b']),
        ],
        'groq-all': [
            CPRunSpec(CPRunType.LLM, config.LLMData.models_by_pname['GROQ.llama-3.3-70b-versatile']),
            CPRunSpec(CPRunType.LLM, config.LLMData.models_by_pname['GROQ.qwen-qwq-32b']),
            CPRunSpec(CPRunType.LLM, config.LLMData.models_by_pname['GROQ.gemma2-9b-it']),
            CPRunSpec(CPRunType.LLM, config.LLMData.models_by_pname['GROQ.deepseek-r1-distill-llama-70b']),
        ],

        'gorbash-test-gg1': [
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.llama3.2:1b'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.llama3.2:3b'], 'gg1', 16384),

            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.huggingface.co/mradermacher/Benchmaxx-Llama-3.2-1B-Instruct-GGUF:latest'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.hf.co/mradermacher/Benchmaxx-Llama-3.2-1B-Instruct-GGUF:F16'], 'gg1', 16384),

            # CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['GROQ.llama-3.3-70b-versatile'], 'gg1'),
            # CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['GROQ.qwen-qwq-32b'], 'gg1'),
            # CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['GITHUB.Phi-4'], 'gg1'),
            # CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['GITHUB.Cohere-command-r-08-2024'], 'gg1'),
            # CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['GITHUB.Llama-3.3-70B-Instruct'], 'gg1'),
            # CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['ANTHROPIC.claude-3-5-haiku-20241022'], 'gg1'),

            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.mistral-nemo:12b'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.mixtral:8x7b'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.gemma2:9b-instruct-fp16'], 'gg1', 16384),
            # CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.gemma2:9b-text-fp16'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.gemma3:1b'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.gemma3:4b'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.gemma3:12b'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.gemma3:12b-it-fp16'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.gemma3:27b'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.gemma3:27b-it-fp16'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.llama3.3:70b'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.llama3.3:70b-instruct-q2_K'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.deepseek-r1:32b'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.deepseek-v2:16b'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.qwq:latest'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.phi4:14b'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.phi4:14b-q8_0'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.phi4:14b-fp16'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.olmo2:13b'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.command-r7b:latest'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.openthinker:32b'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.qwen3:14b-q8_0'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.qwen3:30b-a3b'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.qwen3:30b-a3b-q4_K_M'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.qwen3:32b-q4_K_M'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.qwen3:32b'], 'gg1', 16384),
        ],

        'gorbash-test-fast-ones-gg1': [
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.llama3.2:3b'], 'gg1', 16384),

            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.huggingface.co/mradermacher/Benchmaxx-Llama-3.2-1B-Instruct-GGUF:latest'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.hf.co/mradermacher/Benchmaxx-Llama-3.2-1B-Instruct-GGUF:F16'], 'gg1', 16384),

            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.gemma3:12b'], 'gg1', 16384),
            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.gemma3:27b'], 'gg1', 16384),

            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.phi4:14b'], 'gg1', 16384),

            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.command-r7b:latest'], 'gg1', 16384),
        ],

        'gorbash-test-kf-gg1': [
            # CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.llama3.2:1b'], 'gg1', 16384),
            # CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.llama3.2:3b'], 'gg1', 16384),

            # CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.huggingface.co/mradermacher/Benchmaxx-Llama-3.2-1B-Instruct-GGUF:latest'], 'gg1', 16384),
            # CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.hf.co/mradermacher/Benchmaxx-Llama-3.2-1B-Instruct-GGUF:F16'], 'gg1', 16384),

            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.command-r7b:latest'], 'gg1', 16384),
        ],

        'gorbash-test-dorrit-kf': [

            CPRunSpec(CPRunType.RAG, config.LLMData.models_by_pname['OLLAMA.command-r7b:latest'], 'gg1-dorrit', 16384),
        ],
    }

    ##############################################################
    # llm settings
    ##############################################################
    class LLMRawSettings(LLMSettings):
        def __init__(self, init_n: int, init_temp: float, init_top_p: float, init_max_tokens: int,
                     init_seed: int, init_ctx: int,
                     init_system_message_name: str):
            """
            union of settings for LLMs
            :param init_n:
            :param init_temp:
            :param init_top_p:
            :param init_max_tokens:
            :param init_seed:
            :param init_ctx:
            :param init_system_message_name:

            """
            super().__init__()
            self.n = init_n
            self.temp = init_temp
            self.top_p = init_top_p
            self.max_tokens = init_max_tokens
            self.seed = init_seed
            self.ctx = init_ctx
            self.system_message_name = init_system_message_name
            self.system_message = config.LLMData.sysmsg_all[init_system_message_name]

        def __repr__(self) -> str:
            return f'{self.__class__!s}:{self.__dict__!r}'
            # return f'{self.__class__!s}:{self.result_id=!r},{self.metrics=!r},self.content="{self.content[0:20]}{"..." if len(self.content) > 20 else ""}"'

        def numbers_oneline_logging_str(self) -> str:
            return f'n:{self.n},temp:{self.temp},top_p:{self.top_p},max_tokens:{self.max_tokens}'

        def texts_oneline_logging_str(self) -> str:
            return f'sysmsg:{self.system_message}'

        def specs(self) -> list[BaseSettings.SettingsSpec]:
            sysmsg_names = [key for key in config.LLMData.sysmsg_all]
            return [
                BaseSettings.SettingsSpec(label='n', options=[i for i in range(1, 10)], value=self.n, tooltip='number of results per query'),
                BaseSettings.SettingsSpec(label='temp', options=[float(t) / 10.0 for t in range(0, 21)], value=self.temp, tooltip='responses: 0=very predictable, 2=very random/creative'),
                BaseSettings.SettingsSpec(label='top_p', options=[float(t) / 10.0 for t in range(0, 11)], value=self.top_p, tooltip='responses: 0=less random, 1 more random'),
                BaseSettings.SettingsSpec(label='max_tokens', options=[80, 200, 400, 800, 1000, 1500, 2000], value=self.max_tokens, tooltip='max tokens in response'),
                BaseSettings.SettingsSpec(label='seed', options=[0, 27, 42], value=self.seed, tooltip='random number generator seed'),
                BaseSettings.SettingsSpec(label='ctx', options=[0, 2048, 4096, 8192, 16384, 32768, 65536], value=self.ctx,
                                          tooltip='size of the context window'),
                BaseSettings.SettingsSpec(label='system_message_name', options=sysmsg_names, value=self.system_message_name, tooltip='system/setup text sent with each prompt'),
            ]

        async def change(self, label: str, value: any) -> None:
            if label == 'n':
                self.n = value
            elif label == 'temp':
                self.temp = value
            elif label == 'top_p':
                self.top_p = value
            elif label == 'max_tokens':
                self.max_tokens = value
            elif label == 'seed':
                self.seed = value
            elif label == 'ctx':
                self.ctx = value
            elif label == 'system_message_name':
                self.system_message_name = value
                self.system_message = config.LLMData.sysmsg_all[value]
            else:
                raise ValueError(f'bad label! {label}')

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
                           init_seed=0, init_ctx=2048,
                           init_system_message_name='empty'),
        ],
        'quick': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=40,
                           init_seed=0, init_ctx=2048,
                           init_system_message_name='empty'),
        ],
        'std4': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=800,
                           init_seed=0, init_ctx=2048,
                           init_system_message_name='empty'),
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=400,
                           init_seed=0, init_ctx=2048,
                           init_system_message_name='empty'),
            LLMRawSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=800,
                           init_seed=0, init_ctx=2048,
                           init_system_message_name='empty'),
            LLMRawSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=400,
                           init_seed=0, init_ctx=2048,
                           init_system_message_name='empty'),
        ],
        'gorbash-test': [
            LLMRawSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=800,
                           init_seed=0, init_ctx=2048,
                           init_system_message_name='professional800'),
        ],
        'ollama-warmup': [
            LLMRawSettings(init_n=1, init_temp=1.0, init_top_p=1.0, init_max_tokens=800,
                           init_seed=0, init_ctx=2048,
                           init_system_message_name='carl-sagan'),
        ]
    }

    ##############################################################
    # prompt sets
    ##############################################################

    wakeup_prompt = 'wake up'
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
        'wakeup': [
            [LLMMessagePair('user', wakeup_prompt)],
        ],
        'galaxies': [
            [LLMMessagePair('user', galaxies_prompt)],
        ],
        'galaxies4': [
            [LLMMessagePair('user', galaxies_prompt)],
            [LLMMessagePair('user', galaxies_prompt)],
            [LLMMessagePair('user', galaxies_prompt)],
            [LLMMessagePair('user', galaxies_prompt)],
        ],
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
        'gorbash-compliance-hotline': [
            [LLMMessagePair('user', 'gorbash compliance hotline number?')]
        ],
        'gorbash-security': [
            # ''
            [LLMMessagePair('user', 'what data security does gorbash have?')]
        ],
        'benchmark-prompts': [
            [LLMMessagePair('user', 'Argue for and against the use of kubernetes in the style of a haiku.'), ],
            [LLMMessagePair('user', "Give two concise bullet-point arguments against the Münchhausen trilemma (don't explain what it is)"), ],
            [LLMMessagePair('user',
                            "I went to the market and bought 10 apples. I gave 2 apples to the neighbor and 2 to the repairman. I then went and bought 5 more apples and ate 1. I also gave 3 bananas to my brother. How many apples did I remain with?"), ],
            [LLMMessagePair('user', "Sally (a girl) has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?"), ],
            [LLMMessagePair('user', "Sally (a girl) has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have? Let's think step by step."), ],
            [LLMMessagePair('user', "Explain in a short paragraph quantum field theory to a high-school student."), ],
            [LLMMessagePair('user', "Is Taiwan an independent country?"), ],
            [LLMMessagePair('user', 'Translate this to French, you can take liberties so that it sounds nice: "blossoms paint the spring, nature’s rebirth brings delight and beauty fills the air."'), ],
            [LLMMessagePair('user', "Extract the name of the vendor from the invoice: PURCHASE #0521 NIKE XXX3846. Reply with only the name."), ],
            [LLMMessagePair('user',
                            'Help me find out if this customer review is more "positive" or "negative".\nQ: This movie was watchable but had terrible acting.\nA: negative\nQ: The staff really left us our privacy, we’ll be back.\nA:'), ],
            [LLMMessagePair('user',
                            'What are the 5 planets closest to the sun? Reply with only a valid JSON array of objects formatted like this:\n```\n[{\n"planet": string,\n"distanceFromEarth": number,\n"diameter": number,\n"moons": number\n}]\n```'), ],
            [LLMMessagePair('user', 'Give me the SVG code for a smiley. It should be simple. Reply with only the valid SVG code and nothing else.'), ],
            [LLMMessagePair('user', 'Tell a joke about going on vacation.'), ],
            [LLMMessagePair('user', 'Write a 12-bar blues chord progression in the key of E'), ],
            [LLMMessagePair('user', 'Write me a product description for a 100W wireless fast charger for my website.'), ],
            [LLMMessagePair('user', 'Explain antibiotics'), ],
            [LLMMessagePair('user',
                            'Antibiotics are a type of medication used to treat bacterial infections. They work by either killing the bacteria or preventing them from reproducing, allowing the body’s immune system to fight off the infection. Antibiotics are usually taken orally in the form of pills, capsules, or liquid solutions, or sometimes administered intravenously. They are not effective against viral infections, and using them inappropriately can lead to antibiotic resistance.\nExplain the above in one sentence:'), ],
            [LLMMessagePair('user',
                            'Author-contribution statements and acknowledgements in research papers should state clearly and specifically whether, and to what extent, the authors used AI technologies such as ChatGPT in the preparation of their manuscript and analysis. They should also indicate which LLMs were used. This will alert editors and reviewers to scrutinize manuscripts more carefully for potential biases, inaccuracies and improper source crediting. Likewise, scientific journals should be transparent about their use of LLMs, for example when selecting submitted manuscripts.\nMention the large language model based product mentioned in the paragraph above:'), ],
            [LLMMessagePair('user',
                            'Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer.\nContext: Teplizumab traces its roots to a New Jersey drug company called Ortho Pharmaceutical. There, scientists generated an early version of the antibody, dubbed OKT3. Originally sourced from mice, the molecule was able to bind to the surface of T cells and limit their cell-killing potential. In 1986, it was approved to help prevent organ rejection after kidney transplants, making it the first therapeutic antibody allowed for human use.\nQuestion: What was OKT3 originally sourced from?'), ],
            [LLMMessagePair('user', 'Classify the text into neutral, negative or positive.\nText: I think the food was okay.\nSentiment:'), ],
            [LLMMessagePair('user', 'Classify the text into neutral, negative or positive.\nText: I think the vacation is okay.\nSentiment: neutral\nText: I think the food was okay.\nSentiment:'), ],
            [LLMMessagePair('user', 'Classify the text into nutral, negative or positive.\nText: I think the vacation is okay.\nSentiment:'), ],
            [LLMMessagePair('user',
                            'The following is a conversation with an AI research assistant. The assistant tone is technical and scientific.\nHuman: Hello, who are you?\nAI: Greeting! I am an AI research assistant. How can I help you today?\nHuman: Can you tell me about the creation of blackholes?\nAI:'), ],
            [LLMMessagePair('user',
                            'The following is a conversation with an AI research assistant. The assistant answers should be easy to understand even by primary school students.\nHuman: Hello, who are you?\nAI: Greeting! I am an AI research assistant. How can I help you today?\nHuman: Can you tell me about the creation of black holes?\nAI:'), ],
            [LLMMessagePair('user', 'What is 9,000 * 9,000?'), ],
            [LLMMessagePair('user', 'The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1.\nA:'), ],
            [LLMMessagePair('user', 'The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1.'), ],
            [LLMMessagePair('user', 'Solve by breaking the problem into steps. First, identify the odd numbers, add them, and indicate whether the result is odd or even.'), ],
        ]
    }
