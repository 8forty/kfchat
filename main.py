import logging
import os
import time
import traceback

from fastapi import FastAPI
from nicegui import ui
from typing_extensions import OrderedDict

import chromadbpage
import config
import logstuff
from chatpage import chatpage
from config import FTSType
from llmconfig import llmconfig_factory
from llmconfig.llm_anthropic_config import LLMAnthropicSettings
from llmconfig.llm_ollama_config import LLMOllamaSettings
from llmconfig.llm_openai_config import LLMOpenAISettings
from llmconfig.llmconfig import LLMConfig
from vectorstore import vsapi_factory
from vectorstore.vschroma_settings import VSChromaSettings

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

app = FastAPI()


@app.get("/kfchatroot")
async def root():
    return {"message": "Hello kfchat"}


def init_with_fastapi(fastapi_app: FastAPI) -> None:
    log.info('init_with_fastapi')
    ui.run_with(fastapi_app, storage_secret='pick your private secret here', favicon='xpluto.jpg', title=config.name)

    # setup llm
    # todo: these should come from somewhere, e.g. pref screen
    # todo: init_n: openai,azure,gemini:any(?) value works; groq: requires 1;
    settings = {'openai': LLMOpenAISettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=800, init_system_message_name='professional800'),
                'anthropic': LLMAnthropicSettings(init_temp=0.7, init_top_p=1.0, init_max_tokens=800, init_system_message_name='professional800'),
                # todo? ollama: only 1 resp for any value of n?
                'ollama': LLMOllamaSettings(init_temp=0.7, init_top_p=1.0, init_max_tokens=800, init_seed=0, init_ctx=2048, init_system_message_name='professional800')}
    llm_configs_list: list[LLMConfig] = []
    for model_spec in config.llm_data.models:
        llm_configs_list.append(llmconfig_factory.create_one(model_spec=model_spec, settings=settings))
    all_llm_configs = OrderedDict({f'{lc.provider()}.{lc.model_name}': lc for lc in llm_configs_list})
    init_llm = 'GITHUB.gpt-4o'

    # setup vs
    try:
        retry_wait_seconds = 15
        while True:
            try:
                # todo: configure these
                vssettings = VSChromaSettings(init_n=2, init_fts_type=FTSType.SQLITE3_TRIGRAM_IMPROVED)
                vsparms = config.env.copy()
                vectorstore = vsapi_factory.create_one('chroma', vssettings, parms=vsparms)  # todo: add to env
                vectorstore.warmup()
                break
            except (Exception,) as e:
                if str(e).startswith('Could not connect'):
                    print(f'!!! Chroma client connect error, will retry in {retry_wait_seconds} secs: {e.__class__.__name__}: {e}')
                    time.sleep(retry_wait_seconds)  # todo: configure this?
                    continue
                raise

    except (Exception,) as e:
        log.warning(f'ERROR making vector-store client objects: {e.__class__.__name__}: {e}')
        exc = traceback.format_exc()  # sys.exc_info())
        log.warning(f'{exc}')
        raise

    # the chat page
    cp = chatpage.ChatPage(all_llm_configs=all_llm_configs, init_llm_name=init_llm, vectorstore=vectorstore, parms=config.env)
    cp.setup('/', 'Chat')

    # the chromadb page
    chromadbpage.setup('/chromadb', 'ChromaDB Page', vectorstore, config.env)  # todo: enforce VSChroma vectorstore here


def run():
    log.info(f'MATPLOTLIB is {os.getenv("MATPLOTLIB")}')
    init_with_fastapi(app)


run()
# python launch is doing __main__ and uvicorn.run(...)
# if __name__ == "__main__":
# uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
