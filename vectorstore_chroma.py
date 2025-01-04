import logging
import uuid

import chromadb
import langchain_core.globals as lcglobals
from chromadb.errors import InvalidCollectionException
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

import logstuff
from chatexchanges import VectorStoreResponse, VectorStoreResult
from vectorstorebase import VectorStoreBase

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

# langchain settings
lcglobals.set_debug(True)
lcglobals.set_verbose(True)


class VectorStoreChroma(VectorStoreBase):
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, chroma_client: chromadb.ClientAPI, env_values: dict[str, str]):
        super().__init__(env_values)
        self.chroma_client: chromadb.ClientAPI = chroma_client
        self.env_values: dict[str, str] = env_values

    def list_collection_names(self) -> list[str]:
        return [c.name for c in self.chroma_client.list_collections()]

    def ingest_pdf(self, pdf_file_path: str, pdf_name: str):
        # chroma collection name:
        # (1) contains 3-63 characters,
        # (2) starts and ends with an alphanumeric character,
        # (3) otherwise contains only alphanumeric characters, underscores or hyphens (-) [NO SPACES!],
        # (4) contains no two consecutive periods (..) and
        # (5) is not a valid IPv4 address
        collection_name = pdf_name.replace(' ', '-')
        collection_name = collection_name.replace('..', '._')
        if len(collection_name) > 63:
            collection_name = collection_name[:63]
            log.warning(f'collection name too long, shortened')

        if collection_name != pdf_name:
            log.info(f'collection name [{pdf_name}] modified for chroma restrictions to [{collection_name}]')

        # create the collection
        # todo: configure this
        collection = self.chroma_client.create_collection(
            name=collection_name,
            configuration=None,
            metadata=None,
            embedding_function=SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2'),  # default: 'all-MiniLM-L6-v2'
            data_loader=None,
            get_or_create=False
        )

        # load the PDF into LC Document's (qsa: Doc = page, 48 of each)
        docs: list[Document] = PyPDFLoader(file_path=pdf_file_path).load()

        # split into chunks, also LC Document's (qsa/1024/100: 135 chunks for 48 pages)
        chunks = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100).split_documents(docs)

        # remove complex metadata not supported by ChromaDB, pull out the the content as a str
        chunks = [c.page_content for c in filter_complex_metadata(chunks)]

        # create Chroma vectorstore from the chunks
        log.debug(f'adding {len(chunks)} chunks to {collection_name}')
        collection.add(documents=chunks, ids=[str(uuid.uuid4()) for _ in range(0, len(chunks))])

    # todo: get the model and parameters right for the post-response info
    def ask(self, prompt: str, collection_name: str) -> VectorStoreResponse:
        try:
            collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2'),  # default: 'all-MiniLM-L6-v2'
                data_loader=None,
            )
        except InvalidCollectionException:
            errmsg = f'bad collection name: {collection_name}'
            log.warning(errmsg)
            raise ValueError(errmsg)

        query_results = collection.query(
            query_texts=[prompt],
            n_results=2
        )

        # todo: only one query in the prompt?
        vs_results: list[VectorStoreResult] = []
        for prompt_idx in range(0, len(query_results['ids'])):
            for result_idx in range(0, len(query_results['ids'][prompt_idx])):
                metrics = {'distance': query_results['distances'][prompt_idx][result_idx]}
                vs_results.append(VectorStoreResult(query_results['ids'][prompt_idx][result_idx], metrics, query_results['documents'][prompt_idx][result_idx]))

        return VectorStoreResponse(vs_results)

        # vector_store = Chroma(client=self.chroma_client,
        #                       collection_name=collection_name,
        #                       embedding_function=FastEmbedEmbeddings())  # todo: there are a LOT of choices, this is Qdrant FastEmbed
        #
        # # todo: configure this, LOTS of dials in retriever
        # retriever = vector_store.as_retriever(
        #     search_type="similarity_score_threshold",
        #     search_kwargs={"k": 10, "score_threshold": 0.0},
        # )
        #
        # retriever.invoke(prompt)
        #
        # chain = (
        #         {"context": retriever, "question": RunnablePassthrough()}
        #         | self.prompt
        #         | self.model
        #         | StrOutputParser()
        # )
        #
        # return chain.invoke(prompt)
