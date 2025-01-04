import logging

import logstuff
from chatexchanges import VectorStoreResponse

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class VectorStoreBase:

    def __init__(self, env_values: dict[str, str]):
        self.env_values: dict[str, str] = env_values

    def list_collection_names(self) -> list[str]:
        pass

    def ingest_pdf(self, pdf_file_path: str, pdf_name: str):
        pass

    def ask(self, prompt: str, collection_name: str) -> VectorStoreResponse:
        pass
