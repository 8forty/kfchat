import logging
import time
import traceback

from fastapi import FastAPI
from nicegui import ui
from typing_extensions import OrderedDict

import chromadbpage
import config
import logstuff
from llmconfig.llm_anthropic_config import LLMAnthropicSettings
from vectorstore import vsapi_factory
from chatpage import chatpage
from llmconfig.llmoaiconfig import LLMOaiConfig, LLMOaiSettings

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
    settings_openai = LLMOaiSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=800, init_system_message_name='professional800')
    settings_anthropic = LLMAnthropicSettings(init_n=1, init_temp=0.7, init_top_p=1.0, init_max_tokens=800, init_system_message_name='professional800')
    llm_configs_list = [

        LLMOaiConfig(model_name='gpt-4o-mini', provider='github', settings=settings_openai),
        LLMOaiConfig(model_name='gpt-4o', provider='github', settings=settings_openai),
        LLMOaiConfig(model_name='deepseek-r1', provider='github', settings=settings_openai),
        LLMOaiConfig(model_name='Phi-4', provider='github', settings=settings_openai),
        LLMOaiConfig(model_name='AI21-Jamba-1.5-Large', provider='github', settings=settings_openai),
        LLMOaiConfig(model_name='Cohere-command-r-08-2024', provider='github', settings=settings_openai),
        LLMOaiConfig(model_name='Cohere-command-r-plus-08-2024', provider='github', settings=settings_openai),
        LLMOaiConfig(model_name='Llama-3.3-70B-Instruct', provider='github', settings=settings_openai),
        LLMOaiConfig(model_name='Mistral-Large-2411', provider='github', settings=settings_openai),

        LLMOaiConfig(model_name='llama-3.3-70b-versatile', provider='groq', settings=settings_openai),
        LLMOaiConfig(model_name='deepseek-r1-distill-llama-70b', provider='groq', settings=settings_openai),
        LLMOaiConfig(model_name='gemma2-9b-it', provider='groq', settings=settings_openai),
        LLMOaiConfig(model_name='mixtral-8x7b-32768', provider='groq', settings=settings_openai),

        LLMOaiConfig(model_name='llama3.2:1b', provider='ollama', settings=settings_openai),
        LLMOaiConfig(model_name='llama3.2:3b', provider='ollama', settings=settings_openai),
        LLMOaiConfig(model_name='mistral-nemo:12b', provider='ollama', settings=settings_openai),
        LLMOaiConfig(model_name='gemma2:2b', provider='ollama', settings=settings_openai),
        LLMOaiConfig(model_name='gemma2:9b', provider='ollama', settings=settings_openai),
        LLMOaiConfig(model_name='llama3.3:70b-instruct-q2_K', provider='ollama', settings=settings_openai),
        LLMOaiConfig(model_name='deepseek-r1:1.5b', provider='ollama', settings=settings_openai),
        LLMOaiConfig(model_name='deepseek-r1:8b', provider='ollama', settings=settings_openai),
        LLMOaiConfig(model_name='deepseek-r1:14b', provider='ollama', settings=settings_openai),
        LLMOaiConfig(model_name='deepseek-v2:16b', provider='ollama', settings=settings_openai),
        LLMOaiConfig(model_name='phi4:14b', provider='ollama', settings=settings_openai),

        LLMOaiConfig(model_name='gemini-1.5-flash', provider='gemini', settings=settings_openai),
        LLMOaiConfig(model_name='gemini-1.5-flash-8b', provider='gemini', settings=settings_openai),
        LLMOaiConfig(model_name='gemini-1.5-pro', provider='gemini', settings=settings_openai),
        LLMOaiConfig(model_name='gemini-2.0-flash-001', provider='gemini', settings=settings_openai),
        LLMOaiConfig(model_name='gemini-2.0-flash-lite-preview-02-05', provider='gemini', settings=settings_openai),
        LLMOaiConfig(model_name='gemini-2.0-pro-exp-02-05', provider='gemini', settings=settings_openai),
        LLMOaiConfig(model_name='gemini-2.0-flash-thinking-exp-01-21', provider='gemini', settings=settings_openai),

        LLMOaiConfig(model_name='gpt-4o-mini', provider='openai', settings=settings_openai),
        LLMOaiConfig(model_name='gpt-4o', provider='openai', settings=settings_openai),

        LLMOaiConfig(model_name='RFI-Automate-GPT-4o-mini-2000k',  # really the deployment name for azure
                     provider='azure', settings=settings_openai),

        # LLMAnthropicConfig(model_name='claude-3-5-haiku-20241022', provider_name='anthropic', settings=settings_anthropic),
        # LLMAnthropicConfig(model_name='claude-3-5-sonnet-20241022', provider_name='anthropic', settings=settings_anthropic),
    ]
    llm_configs = OrderedDict({f'{lc._provider}.{lc.model_name}': lc for lc in llm_configs_list})
    init_llm = 'github.gpt-4o'

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
                print(f'!!! Chroma client error, will retry in {retry_wait_seconds} secs: {e.__class__.__name__}: {e}')
            time.sleep(retry_wait_seconds)  # todo: configure this?
    except (Exception,) as e:
        log.warning(f'ERROR making vector-store client objects: {e}')
        exc = traceback.format_exc()  # sys.exc_info())
        log.warning(f'{exc}')
        raise

    # the chat page
    cp = chatpage.ChatPage(llm_configs=llm_configs, init_llm=init_llm, vectorstore=vectorstore, parms=config.env)
    cp.setup('/', 'Chat')

    # the chromadb page
    chromadbpage.setup('/chromadb', 'ChromaDB Page', vectorstore, config.env)  # todo: enforce VSChroma vectorstore here


def run():
    init_with_fastapi(app)


run()
# python launch is doing __main__ and uvicorn.run(...)
# if __name__ == "__main__":
# uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
