from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter, NLTKTextSplitter, SpacyTextSplitter

chunkers = {
    RecursiveCharacterTextSplitter.__name__: {'function': RecursiveCharacterTextSplitter},
    SemanticChunker.__name__: {'function': SemanticChunker},
    NLTKTextSplitter.__name__: {'function': NLTKTextSplitter},
    SpacyTextSplitter.__name__: {'function': SpacyTextSplitter},
}
