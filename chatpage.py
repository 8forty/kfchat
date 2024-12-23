import builtins
import logging
import sys
import timeit
import traceback

from fastapi import Request
from nicegui import ui, run
from nicegui.elements.input import Input
from nicegui.elements.spinner import Spinner
from openai.types.chat import ChatCompletion

import chat
import config
import frame
import logstuff
import vectorstore_chroma
from vectorstore_chroma import VectorStoreChroma

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMConfig:
    def __init__(self):
        # todo: these should come from e.g. pref screen
        self.model_name: str = ['llama3.2:1b', 'llama3.2:3b', 'llama3.3:70b', 'qwen2.5:0.5b', 'gemma2:2b', 'qwq'][1]
        self.temp: float = 0.7
        self.max_tokens: int = 80
        self.system_message: str = (f'You are a helpful chatbot that talks in a conversational manner. '
                                    f'Your responses must always be less than {self.max_tokens} tokens.')


class InstanceData:

    def __init__(self, api_type: str):
        self.general_chat_value: str = '<general chat>'
        self.exchanges: chat.ChatExchanges = chat.ChatExchanges(config.chat_exchanges_circular_list_count)
        self.chat = chat.Chat(api_type)
        self.chat_source_name: str = self.general_chat_value
        self.chat_source: VectorStoreChroma | None = None

    def change_chat_source(self, source_name: str):
        if source_name == self.general_chat_value:
            self.chat_source = None
        else:
            self.chat_source = VectorStoreChroma(vectorstore_chroma.chromadb_client)
        self.chat_source_name = source_name

    @ui.refreshable
    async def refresh_chat_exchanges(self, llm_config: LLMConfig) -> None:
        vectorstore_chroma.setup_once()

        # the configuration selects
        with (ui.row().classes('w-full border-solid border border-black place-content-center')):
            collections: list[str] = [self.general_chat_value]
            colname_list: list[str] = [c.name for c in vectorstore_chroma.chromadb_client.list_collections()]
            colname_list.sort()
            for colname in colname_list:
                collections.append(colname)
            ui.select(label='Chat Source:',
                      options=collections,
                      value=self.chat_source_name,
                      ).on_value_change(lambda vc: self.change_chat_source(vc.value)).props('square outlined label-color=green')

        # todo: local-storage-session to separate messages
        if self.exchanges.len() > 0:
            for exchange in self.exchanges.list():

                # the prompt
                ui.label(exchange.prompt).classes('w-full font-bold text-lg text-blue text-left px-10')

                with ui.column().classes('w-full gap-y-0'):
                    # the response(s)
                    context_info = f'{self.exchanges.id()},'
                    completion_extra = ''
                    # todo: metrics, etc.
                    if exchange.completion is not None:
                        context_info += f'{self.chat.model_api_type()}:{llm_config.model_name},{llm_config.temp},{llm_config.max_tokens}]: '
                        completion_extra = f'{exchange.completion.usage.prompt_tokens} / {exchange.completion.usage.completion_tokens}'
                        for choice in exchange.completion.choices:
                            ui.label(f'[c]: {choice.message.content}').classes('w-full font-bold text-lg text-green text-left px-10')

                    if exchange.vector_store_response is not None:
                        for result in exchange.vector_store_response.results:
                            ui.label(f'[v]: {result.content}').classes('w-full font-bold text-lg text-green text-left px-10')
                            ui.label(f'distance: {result.metrics['distance']}').classes('w-full italic text-xs text-black text-left px-10')

                    # the context info
                    ui.label(f'[{context_info}]: '
                             f'{completion_extra}'
                             f'{exchange.response_duration_secs:.1f}s'
                             ).classes('w-full italic text-xs text-black text-left px-10')
                    ui.label(f'{llm_config.system_message}').classes('w-full italic text-xs text-black text-left px-10')

                    # stop problems info
                    stop_problems_string = ''
                    for choice_idx, stop_problem in exchange.stop_problems().items():
                        stop_problems_string += f'stop[{choice_idx}]:{stop_problem}'
                    if len(stop_problems_string) > 0:
                        ui.label(f'{stop_problems_string}').classes('w-full italic text-xs text-red text-left px-10')

                    if exchange.overflowed():
                        ui.label(f'exchange history overflowed!').classes('w-full italic text-xs text-red text-left px-10')
        else:
            ui.label('No messages yet').classes('mx-auto my-36')

        try:
            await ui.context.client.connected()  # run_javascript which is only possible after connecting
        except builtins.TimeoutError:
            log.warning(f'TimeoutError waiting for client connection, connection ignored')
            return
        await ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')


class ChatPage:

    def __init__(self, api_type: str):
        # anything in here is shared by all instances of ChatPage
        self.api_type: str = api_type

        # todo: configure this
        self.llm_config = LLMConfig()

    def setup(self, path: str, pagename: str):

        def do_chat(prompt: str, idata: InstanceData) -> ChatCompletion | None:
            # todo: count tokens, etc.
            completion = idata.chat.chat(self.llm_config.model_name,
                                         temp=self.llm_config.temp, max_tokens=self.llm_config.max_tokens,
                                         n=1,
                                         sysmsg=self.llm_config.system_message,
                                         prompt=prompt,
                                         convo=idata.exchanges)
            return completion

        def do_vector_search(prompt: str, idata: InstanceData):
            vsresponse = idata.chat_source.ask(prompt, idata.chat_source_name)
            return vsresponse

        async def handle_enter_chat(request, prompt_input: Input, spinner: Spinner, idata: InstanceData) -> None:
            prompt = prompt_input.value.strip()
            log.info(f'(exchanges[{idata.exchanges.id()}]) prompt({self.api_type}:{self.llm_config.model_name},{self.llm_config.temp},{self.llm_config.max_tokens}): {prompt}')
            prompt_input.disable()
            logstuff.update_from_request(request)  # updates logging prefix with info from each request

            start = timeit.default_timer()
            spinner.set_visibility(True)

            # todo: file of prompts
            # if prompt.startswith('*'):  # load a file of prompts
            #     with

            completion = None
            try:
                completion = await run.io_bound(do_chat, prompt, idata)
            except (Exception,):
                e = f'{sys.exc_info()[0].__name__}: {sys.exc_info()[1]}'
                traceback.print_exc(file=sys.stdout)
                log.warning(f'chat error! {e}')
                ui.notify(message=f'chat error! {e}', position='top', type='negative', close_button='Dismiss', timeout=0)

            spinner.set_visibility(False)

            if completion is not None:
                ce = chat.ChatExchange(prompt, response_duration_secs=timeit.default_timer() - start,
                                       completion=completion, vector_store_response=None)
                for choice_idx, sp_text in ce.stop_problems().items():
                    log.warning(f'stop problem from prompt {prompt} choice[{choice_idx}]: {sp_text}')
                idata.exchanges.append(ce)

            prompt_input.value = ''
            prompt_input.enable()
            idata.refresh_chat_exchanges.refresh(self.llm_config)
            await prompt_input.run_method('focus')

        async def handle_enter_vector_search(request, prompt_input: Input, spinner: Spinner, idata: InstanceData) -> None:
            prompt = prompt_input.value.strip()
            log.info(f'(exchanges[{idata.exchanges.id()}]) prompt(): {prompt}')
            prompt_input.disable()
            logstuff.update_from_request(request)  # updates logging prefix with info from each request

            start = timeit.default_timer()
            spinner.set_visibility(True)

            # todo: file of prompts
            # if prompt.startswith('*'):  # load a file of prompts
            #     with

            vsresponse = None
            try:
                log.debug(f'vector search with [{idata.chat_source_name}]: {prompt}')
                vsresponse = await run.io_bound(do_vector_search, prompt, idata)
                # todo: put this in an object
                log.debug(f'vector search result[{type(vsresponse)}]: {vsresponse}')
            except (Exception,):
                e = f'{sys.exc_info()[0].__name__}: {sys.exc_info()[1]}'
                traceback.print_exc(file=sys.stdout)
                log.warning(f'vector-search error! {e}')
                ui.notify(message=f'vector-search error! {e}', position='top', type='negative', close_button='Dismiss', timeout=0)

            spinner.set_visibility(False)

            if vsresponse is not None:
                ce = chat.ChatExchange(prompt, response_duration_secs=timeit.default_timer() - start, completion=None, vector_store_response=vsresponse)
                idata.exchanges.append(ce)

            prompt_input.value = ''
            prompt_input.enable()
            idata.refresh_chat_exchanges.refresh(self.llm_config)
            await prompt_input.run_method('focus')

        async def handle_enter(request, prompt_input: Input, spinner: Spinner, idata: InstanceData) -> None:
            if idata.chat_source is None:
                await handle_enter_chat(request, prompt_input, spinner, idata)
            else:
                await handle_enter_vector_search(request, prompt_input, spinner, idata)

        @ui.page(path)
        async def index(request: Request) -> None:
            logstuff.update_from_request(request)
            log.info(f'route triggered')

            idata = InstanceData(self.api_type)

            # the footer is a "top-level" element in nicegui, so need not be setup in visual page order
            # so I create it here to make sure prompt_input exists before it's needed
            with ui.footer().classes('bg-slate-100 h-24'):
                with ui.row().classes('w-full'):
                    spinner = ui.spinner(size='xl')
                    spinner.set_visibility(False)
                    prompt_input = (ui.input(placeholder="Enter prompt")
                                    .classes('flex-grow')
                                    .props('rounded outlined')
                                    .props('color=primary')
                                    .props('bg-color=white')
                                    )
                    prompt_input.on('keydown.enter', lambda req=request, i=idata: handle_enter(req, prompt_input, spinner, i))

            # setup the standard "frame" for all pages
            with frame.frame(f'{config.name} {pagename}', 'bg-white'):
                with ui.column().classes('w-full flex-grow'):  # .classes('w-full max-w-2xl mx-auto items-stretch'):
                    await idata.refresh_chat_exchanges(self.llm_config)

            await prompt_input.run_method('focus')
