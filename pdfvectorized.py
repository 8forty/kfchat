import logging
import os
import time

import chromadb
import dotenv
import langchain_core.globals as lcglobals
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, AzureChatOpenAI
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

# langchain settings
lcglobals.set_debug(True)
lcglobals.set_verbose(True)

dotenv.load_dotenv()


class PDFVectorized:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, chroma_client):

        self.chroma_client = chroma_client

        # todo: configure/arg this
        # self.model = ChatOllama(model='llama3.2:3b')

        self.model = ChatGroq(model_name='llama-3.3-70b-versatile',
                              groq_api_key=os.getenv('GROQ_API_KEY'))
        # groq_api_base=os.getenv('GROQ_ENDPOINT' # only use this if "using a proxy or service emulator"

        # self.model = ChatOpenAI(model_name='llama-3.3-70b-versatile',
        #                         openai_api_key=os.getenv('GROQ_API_KEY'),
        #                         openai_api_base=os.getenv('GROQ_ENDPOINT'))  # huh, endpoint instead of base for the openai emulation in groq

        # self.model = ChatOpenAI(model_name='gpt-4o-mini',
        #                         openai_api_key=os.getenv("OPENAI_API_KEY"),
        #                         openai_api_base=os.getenv('OPENAI_API_BASE'))

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

        self.vector_store = None
        self.retriever = None
        self.chain = None

    # todo: async?
    def ingest(self, pdf_file_path: str, pdf_name: str):
        # collection name:
        # (1) contains 3-63 characters,
        # (2) starts and ends with an alphanumeric character,
        # (3) otherwise contains only alphanumeric characters, underscores or hyphens (-) [NO SPACES!],
        # (4) contains no two consecutive periods (..) and
        # (5) is not a valid IPv4 address
        collection_name = pdf_name.replace(' ', '-')
        collection_name = collection_name.replace('..', '._')
        if len(collection_name) > 63:
            collection_name = collection_name[:63]

        # load the PDF into LC Document's (qsa: Doc = page, 48 of each)
        docs: list[Document] = PyPDFLoader(file_path=pdf_file_path).load()

        # split into chunks, also LC Document's (qsa/1024/100: 135 chunks for 48 pages)
        chunks = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100).split_documents(docs)

        # remove complex metadata not supported by ChromaDB
        chunks = filter_complex_metadata(chunks)

        # todo: configure this
        # create Chroma vectorstore from the chunks
        self.vector_store = Chroma.from_documents(client=self.chroma_client,
                                                  collection_name=collection_name,
                                                  documents=chunks,
                                                  embedding=FastEmbedEmbeddings(),  # todo: there are a LOT of choices, this is Qdrant FastEmbed
                                                  collection_metadata={'pdf_name': pdf_name}
                                                  )

    # todo: get the model and parameters right in the post-response info
    def ask(self, query: str):
        if not self.vector_store:
            self.vector_store = Chroma(client=self.chroma_client,
                                       # collection_name='northwind-2023-Benefits-At-A-Glance.pdf',
                                       # collection_name='citizens-energy-benefits.pdf',
                                       collection_name='oregon.pdf',
                                       embedding_function=FastEmbedEmbeddings())  # todo: there are a LOT of choices, this is Qdrant FastEmbed

        # todo: configure this, LOTS of dials in retriever
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
