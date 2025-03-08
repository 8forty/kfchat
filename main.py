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
from config import FTSType
from llmconfig.llm_anthropic_config import LLMAnthropicSettings, LLMAnthropicConfig
from llmconfig.llmconfig import LLMConfig
from vectorstore import vsapi_factory
from chatpage import chatpage
from llmconfig.llm_openai_config import LLMOpenAIConfig, LLMOpenAISettings
from vectorstore.vschroma_settings import VSChromaSettings
from vectorstore.vssettings import VSSettings

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
    # todo: init_n: openai,azure,gemini:any(?) value works; ollama: only 1 resp for any value; groq: requires 1;
    settings_openai = LLMOpenAISettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=800, init_system_message_name='professional800')
    settings_anthropic = LLMAnthropicSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=800, init_system_message_name='professional800')
    llm_configs_list: list[LLMConfig] = []
    for model_spec in config.LLMData.models:
        if model_spec.api.lower() == 'openai':
            llm_configs_list.append(LLMOpenAIConfig(model_name=model_spec.name, provider=model_spec.provider, settings=settings_openai))
        elif model_spec.api.lower() == 'anthropic':
            llm_configs_list.append(LLMAnthropicConfig(model_name=model_spec.name, provider=model_spec.provider, settings=settings_anthropic))
    all_llm_configs = OrderedDict({f'{lc.provider()}.{lc.model_name}': lc for lc in llm_configs_list})
    init_llm = 'GITHUB.gpt-4o'

    # setup vs
    try:
        retry_wait_seconds = 15
        while True:
            try:
                vssettings = VSChromaSettings(init_n=2, init_fts_type=FTSType.SQLITE3_TRIGRAM_IMPROVED)
                vsparms = config.env.copy()
                vectorstore = vsapi_factory.create_one('chroma', vssettings, parms=vsparms)  # todo: add to env
                vectorstore.warmup()
                break
            except (Exception,) as e:
                print(f'!!! Chroma client error, will retry in {retry_wait_seconds} secs: {e.__class__.__name__}: {e}')
            time.sleep(retry_wait_seconds)  # todo: configure this?
    except (Exception,) as e:
        log.warning(f'ERROR making vector-store client objects: {e}')
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
