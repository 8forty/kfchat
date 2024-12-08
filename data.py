
import os

from dotenv import load_dotenv

load_dotenv(override=True)

warmup_data = {
    'messageset': [
        ('system', 'You are a helpful assistant that talks like Carl Sagan.'),
        ('user', 'How many galaxies are there?'),
    ],
    'temp': 1.0,
    'max_tokens': 800,
}

apis = {
    'ollama': {
        'll1b': ['llama3.2:1b'],

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
            'OLLAMA_ENDPOINT': 'http://localhost:11434/v1/',
        },
    },
    'openai': {
        '4omini': ['gpt-4o-mini'],
        'std3': ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini'],
        'parms': {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        }
    },
    'azure': {
        '4omini': ['RFI-Automate-GPT-4o-mini-2000k'],
        'parms': {
            'AZURE_OPENAI_API_KEY': os.getenv('AZURE_OPENAI_API_KEY'),
            'AZURE_OPENAI_API_VERSION': os.getenv('AZURE_OPENAI_API_VERSION'),
            'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT'),
        }
    }
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

csagan_sysmsg = 'You are a helpful assistant that talks like Carl Sagan.'
generic_sysmsg = 'You are a helpful chatbot that talks in a conversational manner.'
answer_sysmsg = ('You are a chatbot that answers questions that are labeled "Question:" based on the context labeled "Context:". '
                 'Keep your answers short and concise. Always respond "Unsure about answer" if not sure about the answer.')
textclass_sysmsg = 'Classify each prompt into neutral, negative or positive.'
technical_sysmsg = 'You are an AI research assistant. Respond in a tone that is technical and scientific.'

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

csagan_galaxies = {
    'name': 'csagan_galaxies',
    'messages': [('system', csagan_sysmsg), ('user', galaxies_prompt)]
}
explain_antibiotics = {
    'name': 'explain_antibiotics',
    'messages': [('system', generic_sysmsg), ('user', explain_prompt)]
}
one_sentence = {
    'name': 'one_sentence',
    'messages': [('system', generic_sysmsg), ('user', onesent_prompt)]
}
info_extract = {
    'name': 'info_extract',
    'messages': [('system', generic_sysmsg), ('user', info_prompt)]
}
qa = {
    'name': 'qa',
    'messages': [('system', answer_sysmsg), ('user', teplizumab_prompt)]
}
text_classification = {
    'name': 'text_classification',
    'messages': [('system', textclass_sysmsg), ('user', neutralfood_prompt)]
}
technical = {
    'name': 'technical',
    'messages': [('system', technical_sysmsg), ('user', blackholes_prompt)]
}

message_sets = {
    'carl': [csagan_galaxies],
    'space': [csagan_galaxies, technical],
    'text': [text_classification],
    'std7': [csagan_galaxies, explain_antibiotics, one_sentence, info_extract, qa, text_classification, technical],
}
