import logging

from fastapi import FastAPI
from nicegui import ui

import chatpage
import chromadbpage
import config

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(config.logging_level)

app = FastAPI()


@app.get("/kfchatroot")
async def root():
    return {"message": "Hello kfchat"}


def init_with_fastapi(fastapi_app: FastAPI) -> None:
    log.info('init_with_fastapi')
    ui.run_with(fastapi_app, storage_secret='pick your private secret here')

    cp = chatpage.ChatPage('ollama')
    cp.setup('/', 'Chat')

    chromadbpage.setup('/chromadb')


def run():
    init_with_fastapi(app)


run()
# python launch is doing __main__ and uvicorn.run(...
# if __name__ == "__main__":
# uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
