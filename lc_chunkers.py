from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, NLTKTextSplitter, SpacyTextSplitter

from llmoaiconfig import llm_api_types_config

chunkers = {
    f'{RecursiveCharacterTextSplitter.__name__}:1000/200': {'function': RecursiveCharacterTextSplitter, 'args': {'chunk_size': 1000, 'chunk_overlap': 200}},

    f'{SemanticChunker.__name__}:oai(ada002)/defaults': {'function': SemanticChunker,
                                                         'args': {'embeddings': OpenAIEmbeddings(model='text-embedding-ada-002',
                                                                                                 openai_api_key=llm_api_types_config['openai']['key'])}},
    f'{SemanticChunker.__name__}:oai(3-small)/defaults': {'function': SemanticChunker,
                                                          'args': {'embeddings': OpenAIEmbeddings(model='text-embedding-3-small',
                                                                                                  openai_api_key=llm_api_types_config['openai']['key'])}},
    f'{SemanticChunker.__name__}:oai(3-small)/pct,95.0': {'function': SemanticChunker,
                                                          'args': {'embeddings': OpenAIEmbeddings(model='text-embedding-3-small',
                                                                                                  openai_api_key=llm_api_types_config['openai']['key']),
                                                                   'breakpoint_threshold_type': 'percentile',
                                                                   'breakpoint_threshold_amount': 95.0}},
    f'{SemanticChunker.__name__}:oai(3-small)/stdev,3.0': {'function': SemanticChunker,
                                                           'args': {'embeddings': OpenAIEmbeddings(model='text-embedding-3-small',
                                                                                                   openai_api_key=llm_api_types_config['openai']['key']),
                                                                    'breakpoint_threshold_type': 'standard_deviation',
                                                                    'breakpoint_threshold_amount': 3.0}},
    f'{SemanticChunker.__name__}:oai(3-small)/iq,1.5': {'function': SemanticChunker,
                                                        'args': {'embeddings': OpenAIEmbeddings(model='text-embedding-3-small',
                                                                                                openai_api_key=llm_api_types_config['openai']['key']),
                                                                 'breakpoint_threshold_type': 'interquartile',
                                                                 'breakpoint_threshold_amount': 1.5}},
    f'{SemanticChunker.__name__}:oai(3-small)/grad,95.0': {'function': SemanticChunker,
                                                           'args': {'embeddings': OpenAIEmbeddings(model='text-embedding-3-small',
                                                                                                   openai_api_key=llm_api_types_config['openai']['key']),
                                                                    'breakpoint_threshold_type': 'gradient',
                                                                    'breakpoint_threshold_amount': 95.0}},
    f'{NLTKTextSplitter.__name__}:defaults': {'function': NLTKTextSplitter, 'args': {}},
    f'{SpacyTextSplitter.__name__}:defaults': {'function': SpacyTextSplitter, 'args': {}},
}
