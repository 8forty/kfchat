import logging

import chromadb
from fastapi import Request
from nicegui import ui

import config
import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(config.logging_level)


def setup(path: str):
    @ui.page(path)
    async def index(request: Request) -> None:
        logstuff.update_from_request(request)
        log.debug(f'chromadbpage route triggered')
        ui.label('chromadb page')

        client = await chromadb.AsyncHttpClient(host='localhost', port=8888)
        for coll in await client.list_collections():
            ui.label(f'collection [{coll.name}]:{coll.metadata}')
            results = await coll.query(
                query_texts=["This is a query document about hawaii"],  # Chroma will embed this for you
                n_results=2  # how many results to return
            )
            ui.label(f'results: {results}')
