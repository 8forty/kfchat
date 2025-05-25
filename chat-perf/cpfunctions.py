import logging
import sys
import traceback

import config
import util
from chatexchanges import VectorStoreResponse, ChatExchange
from llmconfig.llmconfig import LLMConfig
from llmconfig.llmexchange import LLMExchange, LLMMessagePair
from sqlitedata import FTSType
from vectorstore import vsapi_factory
from vectorstore.vschroma_settings import VSChromaSettings

logging.disable(logging.INFO)


class CPFunctions:

    @staticmethod
    def run_llm_prompt(prompt_set: list[LLMMessagePair], context: list[str] | None, llm_config: LLMConfig,
                       all_start: float) -> LLMExchange | None:
        # todo: suppress + note actually allowed parameters
        # log.info(
        #     f'(exchanges[{idata.chat_exchange_id()}]) prompt({idata.mode()}:{idata.llm_config().provider()}:{idata.llm_config().model_name},'
        #     f'{idata.llm_config().settings().numbers_oneline_logging_str()}): "{prompt}"')

        exchange: LLMExchange = llm_config.chat_messages(messages=prompt_set, context=context)
        return exchange

    @staticmethod
    def run_rag_prompt(prompt_set: list[LLMMessagePair], collection_name: str, llm_config: LLMConfig,
                       max_results: int, dense_weight: float, all_start: float) -> LLMExchange | None:
        # log.info(f'(exchanges[{idata.chat_exchange_id()}]) prompt({idata.mode()}:{idata.source()}): "{prompt}"')

        # todo: configure
        vssettings = VSChromaSettings(init_n=1, init_fts_type=FTSType.SQLITE3_TRIGRAM_IMPROVED)
        vsparms = config.env.copy()
        vectorstore = vsapi_factory.create_one('chroma', vssettings, parms=vsparms)  # todo: add to env
        vectorstore.warmup()

        # first get the vector results
        try:
            prompt = ''.join([p.content for p in prompt_set if p.role == 'user'])

            vectorstore.set_collection(collection_name)
            # todo: configure max_results and dense_weight, VSSettings maybe?
            vsresponse: VectorStoreResponse = vectorstore.hybrid_search(query=prompt, max_results=max_results, dense_weight=dense_weight)
            # log.debug(f'rag vector-search response: {vsresponse}')
            context = [r.content for r in vsresponse.results]
            context_len = sum(len(s) for s in context)
            print(f'{util.secs_string(all_start)}: {collection_name} had {len(context)} hits len: {context_len} from vs prompt: {prompt}')

            # then get llm results
            if len(context) > 0:
                exchange: LLMExchange | None = CPFunctions.run_llm_prompt(prompt_set, context, llm_config, all_start)
                return exchange
            else:
                raise ValueError(f'rag found no context! {collection_name}: {prompt}')

        except (Exception,) as e:
            traceback.print_exc(file=sys.stdout)
            errmsg = f'rag error! {e.__class__.__name__}: {e}'
            raise ValueError(errmsg)
