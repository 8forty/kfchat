from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, PDFMinerLoader

docloaders = {
    PyPDFLoader.__name__: {'function': PyPDFLoader, 'filetype': 'pdf'},
    PyMuPDFLoader.__name__: {'function': PyMuPDFLoader, 'filetype': 'pdf'},
    PDFMinerLoader.__name__: {'function': PDFMinerLoader, 'filetype': 'pdf'},
}
