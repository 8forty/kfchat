import logging
import os
import time

import chromadb
import dotenv
import langchain_core.globals as lcglobals
from langchain_community.vectorstores import Chroma
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


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):

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

        while True:
            try:
                self.chroma_client = chromadb.HttpClient(host='localhost', port=8888)  # todo: configure this
                break
            except (Exception,) as e:
                print(f'!!! Chroma client error, will retry in {15} secs: {e}')
            time.sleep(15)  # todo: configure this

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

    # todo: get the model and parameters right in the post-response info
    def ask(self, query: str):
        if not self.vector_store:
            self.vector_store = Chroma(client=self.chroma_client,
                                       # collection_name='northwind-2023-Benefits-At-A-Glance.pdf',  # todo: arg this
                                       # collection_name='citizens-energy-benefits.pdf',
                                       collection_name='oregon.pdf',
                                       embedding_function=FastEmbedEmbeddings(), )

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
