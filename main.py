import logging
import time
import traceback

import dotenv
from fastapi import FastAPI
from nicegui import ui

import chromadbpage
import config
import logstuff
import vsapi_factory
from chatpage import chatpage
from llmoaiconfig import LLMOaiConfig

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
    max_tokens = 800
    system_message_name = 'technical800'
    llm_configs_list = [
        LLMOaiConfig(name='groq33', api_type_name='groq', parms=env_values, model_name='llama-3.3-70b-versatile',
                     init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=max_tokens, init_system_message_name=system_message_name),
        LLMOaiConfig(name='ollama321b', api_type_name='ollama', parms=env_values, model_name='llama3.2:1b',
                     init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=max_tokens, init_system_message_name=system_message_name),
        LLMOaiConfig(name='ollama323b', api_type_name='ollama', parms=env_values, model_name='llama3.2:3b',
                     init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=max_tokens, init_system_message_name=system_message_name),
        LLMOaiConfig(name='ollamag2-9b', api_type_name='ollama', parms=env_values, model_name='gemma2:9b',
                     init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=max_tokens, init_system_message_name=system_message_name),
        LLMOaiConfig(name='ollamag2-27b', api_type_name='ollama', parms=env_values, model_name='gemma2:27b',
                     init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=max_tokens, init_system_message_name=system_message_name),
        LLMOaiConfig(name='openai4omini', api_type_name='openai', parms=env_values, model_name='gpt-4o-mini',
                     init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=max_tokens, init_system_message_name=system_message_name),
        LLMOaiConfig(name='azurerfi', api_type_name='azure', parms=env_values,
                     model_name='RFI-Automate-GPT-4o-mini-2000k',  # really the deployment name for azure
                     init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=max_tokens, init_system_message_name=system_message_name),
        LLMOaiConfig(name='gemini15', api_type_name='gemini', parms=env_values, model_name='gemini-1.5-flash-8b',
                     init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=max_tokens,
                     init_system_message_name=system_message_name),
    ]
    llm_configs = {lc.name: lc for lc in llm_configs_list}

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
    cp = chatpage.ChatPage(llm_configs=llm_configs, init_llm_name='ollama323b', vectorstore=vectorstore, env_values=env_values)
    cp.setup('/', 'Chat')

    # the chromadb page
    chromadbpage.setup('/chromadb', 'ChromaDB Page', vectorstore, env_values)  # todo: enforce VSChroma vectorstore here


def run():
    init_with_fastapi(app)


run()
# python launch is doing __main__ and uvicorn.run(...)
# if __name__ == "__main__":
# uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
