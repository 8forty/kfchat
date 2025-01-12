import logging
import time
import traceback

import chromadb
import dotenv
from fastapi import FastAPI
from nicegui import ui

import chatpage
import chromadbpage
import config
import logstuff
import vsapi_factory
from llmconfig import LLMConfig
from llmapi import LLMAPI
from llmopenai import LLMOpenai
from vschroma import VSChroma

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

    # setup llm
    # todo: these should come from e.g. pref screen
    max_tokens = 80
    system_message = (f'You are a helpful chatbot that talks in a conversational manner. '
                      f'Your responses must always be less than {max_tokens} tokens.')
    # llm_config: LLMConfig = LLMConfig(LLMOpenai('ollama', parms=env_values), model_name='llama3.2:1b',
    #                                   default_temp=0.7, default_max_tokens=max_tokens, default_system_message=system_message)
    # llm_config: LLMConfig = LLMConfig(LLMOpenai('openai', parms=env_values), model_name='gpt-4o-mini',
    #                                   default_temp=0.7, default_max_tokens=max_tokens, default_system_message=system_message)
    llm_config: LLMConfig = LLMConfig(LLMOpenai('groq', parms=env_values), model_name='llama-3.3-70b-versatile',
                                      default_temp=0.7, default_max_tokens=max_tokens, default_system_message=system_message)
    # llm_config: LLMConfig = LLMConfig(LLMOpenai('azure', parms=env_values),
    #                                   model_name='RFI-Automate-GPT-4o-mini-2000k',  # really the deployment name for azure
    #                                   default_temp=0.7, default_max_tokens=max_tokens, default_system_message=system_message)

    # setup vs
    try:
        retry_wait_seconds = 15
        while True:
            try:
                vectorstore = vsapi_factory.create_one('chroma', env_values)
                vectorstore.warmup()
                break
            except (Exception,) as e:
                print(f'!!! Chroma client error, will retry in {retry_wait_seconds} secs: {e}')
            time.sleep(retry_wait_seconds)  # todo: configure this?
    except (Exception,) as e:
        log.warning(f'ERROR making vector-store client objects: {e}')
        exc = traceback.format_exc()  # sys.exc_info())
        log.warning(f'{exc}')
        raise

    # the chat page
    cp = chatpage.ChatPage(llm_config=llm_config, vectorstore=vectorstore, env_values=env_values)
    cp.setup('/', 'Chat')

    # the chromadb page
    chromadbpage.setup('/chromadb', 'ChromaDB Page', vectorstore, env_values)


def run():
    init_with_fastapi(app)


run()
# python launch is doing __main__ and uvicorn.run(...)
# if __name__ == "__main__":
# uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
