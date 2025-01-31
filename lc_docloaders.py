from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, PDFMinerLoader, TextLoader
from langchain_docling import DoclingLoader

docloaders = {
    PyMuPDFLoader.__name__: {'function': PyMuPDFLoader, 'filetypes': ['pdf']},
    TextLoader.__name__: {'function': TextLoader, 'filetypes': ['txt', 'md']},
    DoclingLoader.__name__: {'function': DoclingLoader, 'filetypes': ['docx', 'pptx', 'html', 'pdf']},
    PyPDFLoader.__name__: {'function': PyPDFLoader, 'filetypes': ['pdf']},
    PDFMinerLoader.__name__: {'function': PDFMinerLoader, 'filetypes': ['pdf']},
}
