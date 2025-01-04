import logging

import dotenv
from fastapi import FastAPI
from nicegui import ui

import chatpage
import chromadbpage
import config
import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

app = FastAPI()

dotenv.load_dotenv(override=True)
env_values: dict[str, str] = dotenv.dotenv_values()


@app.get("/kfchatroot")
async def root():
    return {"message": "Hello kfchat"}


def init_with_fastapi(fastapi_app: FastAPI) -> None:
    log.info('init_with_fastapi')
    ui.run_with(fastapi_app, storage_secret='pick your private secret here', favicon='pluto.jpg', title=config.name)

    max_tokens = 80
    system_message = (f'You are a helpful chatbot that talks in a conversational manner. '
                      f'Your responses must always be less than {max_tokens} tokens.')
    llm_config: config.LLMConfig = config.LLMConfig('ollama', env_values=env_values, model_name='llama3.2:1b',
                                                    default_temp=0.8, max_tokens=max_tokens, system_message=system_message)
    cp = chatpage.ChatPage(llm_config, env_values)
    cp.setup('/', 'Chat')

    chromadbpage.setup('/chromadb', 'ChromaDB Page', env_values)


def run():
    init_with_fastapi(app)


run()
# python launch is doing __main__ and uvicorn.run(...)
# if __name__ == "__main__":
# uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
