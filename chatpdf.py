import logging

import chromadb
import langchain_core.globals as lcglobals
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

lcglobals.set_debug(True)
lcglobals.set_verbose(True)


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, llm_model: str = "llama3.2:3b"):

        self.model = ChatOllama(model=llm_model)

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    "You are a helpful assistant that can answer questions about the PDF document that uploaded by the user. ",
                ),
                (
                    "human",
                    "Here is the document pieces: {context}\nQuestion: {question}",
                ),
            ]
        )
        self.chroma_client = chromadb.HttpClient(host='localhost', port=8888)

        self.vector_store = None
        self.retriever = None
        self.chain = None

    # todo: async?
    def ingest(self, pdf_file_path: str, pdf_name: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        # todo: configure this
        # client = chromadb.AsyncHttpClient(host='localhost', port=8888)
        self.vector_store = Chroma.from_documents(client=self.chroma_client,
                                                  collection_name=pdf_name,
                                                  documents=chunks,
                                                  embedding=FastEmbedEmbeddings(),
                                                  collection_metadata={'pdf_file_path': pdf_file_path, 'pdf_name': pdf_name}
                                                  )
        # self.vector_store = Chroma.from_documents(documents=chunks,
        #                                           embedding=FastEmbedEmbeddings(),
        #                                           persist_directory="chroma_db", )

    def ask(self, query: str):
        if not self.vector_store:
            self.vector_store = Chroma(client=self.chroma_client,
                                       # collection_name='northwind-2023-Benefits-At-A-Glance.pdf',  # todo: arg this
                                       collection_name='citizens-energy-benefits.pdf',
                                       embedding_function=FastEmbedEmbeddings(), )
            # self.vector_store = Chroma(persist_directory="chroma_db",
            #                            embedding_function=FastEmbedEmbeddings(), )

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.0},
        )

        self.retriever.invoke(query)

        self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser()
        )

        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
