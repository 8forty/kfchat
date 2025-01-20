from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document


def pypdf_text(server_pdf_path: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    # load the PDF into LC Document's (qsa: Doc = page, 48 of each)
    docs: list[Document] = PyPDFLoader(file_path=server_pdf_path).load()

    # split into chunks, also LC Document's
    chunks = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(docs)

    # remove complex metadata not supported by ChromaDB, pull out the the content as a str
    return [c.page_content for c in filter_complex_metadata(chunks)]


def pypdf_semantic(server_pdf_path: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    # load the PDF into LC Document's (qsa: Doc = page, 48 of each)
    docs: list[Document] = PyPDFLoader(file_path=server_pdf_path).load()

    raise ValueError('pypdf_semantic NOT IMPLMENTED YET')
