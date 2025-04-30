import logging
import sys
import traceback

import config
from chatexchanges import VectorStoreResponse, ChatExchange
from config import FTSType
from llmconfig.llmconfig import LLMConfig
from llmconfig.llmexchange import LLMExchange, LLMMessagePair
from vectorstore import vsapi_factory
from vectorstore.vschroma_settings import VSChromaSettings

logging.disable(logging.INFO)


async def run_llm_prompt(prompt_set: list[LLMMessagePair], context: list[str] | None, llm_config: LLMConfig) -> LLMExchange | None:
    # todo: suppress + note actually allowed parameters
    # log.info(
    #     f'(exchanges[{idata.chat_exchange_id()}]) prompt({idata.mode()}:{idata.llm_config().provider()}:{idata.llm_config().model_name},'
    #     f'{idata.llm_config().settings().numbers_oneline_logging_str()}): "{prompt}"')

    exchange: LLMExchange = llm_config.chat_messages(messages=prompt_set, context=context)

# def run_rag_prompt(prompt: str, max_results: int, dense_weight: float) -> None:
#     # log.info(f'(exchanges[{idata.chat_exchange_id()}]) prompt({idata.mode()}:{idata.source()}): "{prompt}"')
#
#     vssettings = VSChromaSettings(init_n=2, init_fts_type=FTSType.SQLITE3_TRIGRAM_IMPROVED)
#     vsparms = config.env.copy()
#     vectorstore = vsapi_factory.create_one('chroma', vssettings, parms=vsparms)  # todo: add to env
#     vectorstore.warmup()
#
#     # first get the vector results
#     try:
#         # todo: configure max_results and dense_weight
#         vsresponse: VectorStoreResponse = vectorstore.hybrid_search(query=prompt, max_results=0, dense_weight=0.5)
#         # log.debug(f'rag vector-search response: {vsresponse}')
#         context = [r.content for r in vsresponse.results]
#         if len(context) > 0:
#             # then get llm results
#             exchange: LLMExchange | None = await run_llm_prompt(prompt, context, idata)
#
#             if exchange is not None:
#                 # log.debug(f'rag llm exchange responses: {exchange.responses}')
#                 ce = ChatExchange(prompt, response_duration_secs=exchange.response_duration_secs,
#                                   llm_exchange=exchange, vector_store_response=None,
#                                   source=idata.source(), mode=idata.mode())
#                 for choice_idx, sp_text in ce.problems().items():
#                 # log.warning(f'rag stop problem from prompt {prompt} choice[{choice_idx}]: {sp_text}')
#                 idata.add_chat_exchange(ce)
#             else:
#                 raise ValueError(f'rag got no result from LLM! ({idata.source()})')
#         else:
#             raise ValueError(f'rag found no context! ({idata.source()})')
#
#     except (Exception,) as e:
#         traceback.print_exc(file=sys.stdout)
#         errmsg = f'rag error! {e.__class__.__name__}: {e}'
#         # log.warning(errmsg)
#         # ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)
