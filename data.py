import os

from dotenv import load_dotenv

from llmconfig.llmconfig import LLMConfig

load_dotenv(override=True)

warmup_data = {
    'messageset': [
        ('system', 'You are a helpful assistant that talks like Carl Sagan.'),
        ('user', 'How many galaxies are there?'),
    ],
    'temp': 1.0,
    'max_tokens': 800,
}

# dict[api-name, dict[set-name or 'parms', list-of-model-names or dict[parm-name, parm-value]]]
model_sets = {
    'ollama': {
        'll1b': ['llama3.2:1b'],
        'll3b': ['llama3.2:3b'],
        'll33': ['llama3.3:70b'],
        'mistral7b': ['mistral:7b'],
        'gemma29b': ['gemma2:9b'],

        'qwq': ['qwq'],

        'std8': ['llama3.2:1b', 'llama3.2:3b',
                 'phi3:3.8b', 'phi3.5:3.8b',
                 'gemma2:2b', 'gemma2:27b',
                 'qwen2.5:1.5b', 'qwen2.5:32b'],

        'std-small': ['llama3.2:1b',
                      'phi3:3.8b',
                      'gemma2:2b',
                      'qwen2.5:1.5b'],

        'smallest': ['llama3.2:1b',
                     'qwen2.5:1.5b'],

        'std18': ['llama3.2:1b', 'llama3.2:3b',
                  'phi3:3.8b', 'phi3.5:3.8b', 'phi3:14b',
                  'mistral-small:22b', 'mistral-nemo:12b', 'mistral-large:123b',
                  'gemma2:2b', 'gemma2:9b', 'gemma2:27b',
                  'qwen2.5:0.5b', 'qwen2.5:1.5b', 'qwen2.5:3b', 'qwen2.5:7b', 'qwen2.5:14b', 'qwen2.5:32b', 'qwen2.5:72b',
                  ],
        'parms': {
            'OLLAMA_ENDPOINT': os.getenv('OLLAMA_ENDPOINT'),
        },
    },
    'openai': {
        '4omini': ['gpt-4o-mini'],
        'std3': ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini'],
        'parms': {
            'key': os.getenv('kfOPENAI_API_KEY'),
        }
    },
    'azure': {
        '4omini': ['RFI-Automate-GPT-4o-mini-2000k'],
        'parms': {
            'key': os.getenv('kfAZURE_OPENAI_API_KEY'),
            'AZURE_OPENAI_API_VERSION': os.getenv('AZURE_OPENAI_API_VERSION'),
            'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT'),
        }
    },
    'groq': {
        'll1b': ['llama-3.2-1b-preview'],
        'll33': ['llama-3.3-70b-versatile'],
        'mistral7b': ['mixtral-8x7b-32768'],
        'gemma29b': ['gemma2-9b-it'],
        'parms': {
            'key': os.getenv('kfGROQ_API_KEY'),
            'GROQ_ENDPOINT': os.getenv('GROQ_OPENAI_ENDPOINT'),
        }
    },
    'gemini': {
        'g15flash': ['gemini-1.5-flash'],
        'g15flash002': ['gemini-1.5-flash-002'],
        'g15flash8b': ['gemini-1.5-flash-8b'],
        'g15flash8b001': ['gemini-1.5-flash-8b-001'],
        'g15pro': ['gemini-1.5-pro'],
        'parms': {
            'key': os.getenv('kfGEMINI_API_KEY'),
            'GEMINI_ENDPOINT': os.getenv('GEMINI_OPENAI_ENDPOINT'),
        }
    },
}

settings_sets = {
    '1:800': [
        {'temp': 1.0, 'max_tokens': 800},
    ],
    'quick': [
        {'temp': 1.0, 'max_tokens': 80},
    ],
    'std4': [
        {'temp': 1.0, 'max_tokens': 800},
        {'temp': 1.0, 'max_tokens': 400},
        {'temp': 0.7, 'max_tokens': 800},
        {'temp': 0.7, 'max_tokens': 400},
    ],
}

galaxies_prompt = 'How many galaxies are there?'
explain_prompt = 'Explain antibiotics'
onesent_prompt = ('Antibiotics are a type of medication used to treat bacterial infections. They work by either killing '
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

csagan_galaxies = {
    'name': 'csagan_galaxies',
    'messages': [('system', LLMConfig.csagan_sysmsg), ('user', galaxies_prompt)]
}
explain_antibiotics = {
    'name': 'explain_antibiotics',
    'messages': [('system', LLMConfig.conversational_sysmsg), ('user', explain_prompt)]
}
one_sentence = {
    'name': 'one_sentence',
    'messages': [('system', LLMConfig.conversational_sysmsg), ('user', onesent_prompt)]
}
info_extract = {
    'name': 'info_extract',
    'messages': [('system', LLMConfig.conversational_sysmsg), ('user', info_prompt)]
}
qa = {
    'name': 'qa',
    'messages': [('system', LLMConfig.answer_sysmsg), ('user', teplizumab_prompt)]
}
text_classification = {
    'name': 'text_classification',
    'messages': [('system', LLMConfig.textclass_sysmsg), ('user', neutralfood_prompt)]
}
technical = {
    'name': 'technical',
    'messages': [('system', LLMConfig.technical_sysmsg), ('user', blackholes_prompt)]
}

message_sets: dict[str, list] = {
    'carl': [csagan_galaxies],
    'space': [csagan_galaxies, technical],
    'text': [text_classification],
    'std7': [csagan_galaxies, explain_antibiotics, one_sentence, info_extract, qa, text_classification, technical],
}
