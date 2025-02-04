import logging
import time
import traceback

from fastapi import FastAPI
from nicegui import ui

import chromadbpage
import config
import logstuff
import vsapi_factory
from chatpage import chatpage
from llmoaiconfig import LLMOaiConfig, LLMOaiSettings

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

app = FastAPI()


@app.get("/kfchatroot")
async def root():
    return {"message": "Hello kfchat"}


def init_with_fastapi(fastapi_app: FastAPI) -> None:
    log.info('init_with_fastapi')
    ui.run_with(fastapi_app, storage_secret='pick your private secret here', favicon='pluto.jpg', title=config.name)

    # setup llm
    # todo: these should come from somewhere, e.g. pref screen
    settings = LLMOaiSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=800, init_system_message_name='technical800')
    llm_configs_list = [
        LLMOaiConfig(model_name='llama-3.3-70b-versatile', api_type_name='groq', settings=settings),
        LLMOaiConfig(model_name='deepseek-r1-distill-llama-70b', api_type_name='groq', settings=settings),

        # LLMOaiConfig(model_name='llama3.2:1b', api_type_name='ollama', settings=settings),
        LLMOaiConfig(model_name='mistral-nemo:12b', api_type_name='ollama', settings=settings),
        LLMOaiConfig(model_name='llama3.2:3b', api_type_name='ollama', settings=settings),
        LLMOaiConfig(model_name='gemma2:9b', api_type_name='ollama', settings=settings),
        LLMOaiConfig(model_name='llama3.3:70b-instruct-q2_K', api_type_name='ollama', settings=settings),
        # LLMOaiConfig(model_name='gemma2:27b', api_type_name='ollama', settings=settings),

        LLMOaiConfig(model_name='gpt-4o-mini', api_type_name='openai', settings=settings),

        LLMOaiConfig(model_name='RFI-Automate-GPT-4o-mini-2000k',  # really the deployment name for azure
                     api_type_name='azure', settings=settings),

        LLMOaiConfig(model_name='gemini-1.5-flash', api_type_name='gemini', settings=settings),
        LLMOaiConfig(model_name='gemini-1.5-flash-8b', api_type_name='gemini', settings=settings),
        LLMOaiConfig(model_name='gemini-1.5-pro', api_type_name='gemini', settings=settings),

        LLMOaiConfig(model_name='gpt-4o', api_type_name='github', settings=settings),
        LLMOaiConfig(model_name='gpt-4o-mini', api_type_name='github', settings=settings),
        LLMOaiConfig(model_name='deepseek-r1', api_type_name='github', settings=settings),
        # LLMOaiConfig(model_name='gpt-o1-preview', api_type_name='github', settings=settings),
        # LLMOaiConfig(model_name='openai-o1-preview', api_type_name='github', settings=settings),
        # LLMOaiConfig(model_name='o1-preview', api_type_name='github', settings=settings),
    ]
    llm_configs = {lc.model_name: lc for lc in llm_configs_list}

    # setup vs
    try:
        retry_wait_seconds = 15
        while True:
            try:
                vsparms = config.env.copy()
                vectorstore = vsapi_factory.create_one('chroma', parms=vsparms)  # todo: add to env
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
    cp = chatpage.ChatPage(llm_configs=llm_configs, init_llm_model_name=llm_configs_list[0].model_name, vectorstore=vectorstore, parms=config.env)
    cp.setup('/', 'Chat')

    # the chromadb page
    chromadbpage.setup('/chromadb', 'ChromaDB Page', vectorstore, config.env)  # todo: enforce VSChroma vectorstore here


def run():
    init_with_fastapi(app)


run()
# python launch is doing __main__ and uvicorn.run(...)
# if __name__ == "__main__":
# uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
