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
from chatexchange import ChatExchange, ChatExchanges
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
        self.exchanges: ChatExchanges = ChatExchanges(config.chat_exchanges_circular_list_count)
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
        if self.exchanges is not None and len(self.exchanges) > 0:
            for exchange in self.exchanges:

                # the prompt
                ui.label(exchange.prompt).classes('w-full font-bold text-lg text-blue text-left px-10')

                with (ui.column().classes('w-full gap-y-0')):
                    # the response
                    ui.label(exchange.response).classes('w-full font-bold text-lg text-green text-left px-10')

                    # the context info
                    ui.label(f'[{self.exchanges.id()},{self.chat.model_api_type()}:{llm_config.model_name},{llm_config.temp},{llm_config.max_tokens}]: '
                             f'{exchange.token_counts[0]}/{exchange.token_counts[1]} '
                             f'{exchange.duration_seconds:.1f}s'
                             ).classes('w-full italic text-xs text-black text-left px-10')
                    ui.label(f'{llm_config.system_message}').classes('w-full italic text-xs text-black text-left px-10')

                    # stop problems info
                    stop_problems_string = ''
                    for choice_idx, stop_problem in exchange.stop_problems.items():
                        stop_problems_string += f'stop[{choice_idx}]:{stop_problem}'
                    if len(stop_problems_string) > 0:
                        ui.label(f'{stop_problems_string}').classes('w-full italic text-xs text-red text-left px-10')
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

        self.llm_config = LLMConfig()

    def setup(self, path: str, pagename: str):

        def do_chat(prompt: str, idata: InstanceData) -> ChatCompletion | None:
            # todo: count tokens, etc.
            # todo: just save the convo eh?
            convo: list[chat.ChatExchange] = []
            if idata.exchanges is not None:
                for exchange in idata.exchanges:
                    convo.append(chat.ChatExchange(exchange.prompt, exchange.response))

            return idata.chat.chat_batch(self.llm_config.model_name, temp=self.llm_config.temp, max_tokens=self.llm_config.max_tokens, n=1,
                                         sysmsg=self.llm_config.system_message,
                                         prompt=prompt,
                                         convo=convo)

        def do_vector_search(prompt: str, idata: InstanceData):
            response = idata.chat_source.ask(prompt, idata.chat_source_name)
            return response['documents'][0][0]

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

            response = None
            try:
                response = await run.io_bound(do_chat, prompt, idata)
            except (Exception,):
                e = f'{sys.exc_info()[0].__name__}: {sys.exc_info()[1]}'
                traceback.print_exc(file=sys.stdout)
                log.warning(f'chat error! {e}')
                ui.notify(message=f'chat error! {e}', position='top', type='negative', close_button='Dismiss', timeout=0)

            spinner.set_visibility(False)

            if response is not None:
                if idata.exchanges is not None:
                    idata.exchanges.append(ChatExchange(prompt, response.choices[0].message.content,
                                                        (response.usage.prompt_tokens, response.usage.completion_tokens),
                                                        timeit.default_timer() - start,
                                                        chat.Chat.check_for_stop_problems(response)))
                else:
                    log.warning(f'exchanges list is None!  prompt/response not saved')

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

            response = None
            try:
                log.debug(f'vector search with [{idata.chat_source_name}]: {prompt}')
                response = await run.io_bound(do_vector_search, prompt, idata)
                # todo: put this in an object
                log.debug(f'vector search response[{type(response)}]: {response}')
            except (Exception,):
                e = f'{sys.exc_info()[0].__name__}: {sys.exc_info()[1]}'
                traceback.print_exc(file=sys.stdout)
                log.warning(f'vector-search error! {e}')
                ui.notify(message=f'vector-search error! {e}', position='top', type='negative', close_button='Dismiss', timeout=0)

            spinner.set_visibility(False)

            if response is not None:
                if idata.exchanges is not None:
                    idata.exchanges.append(ChatExchange(prompt, response,
                                                        (-1, -1),
                                                        timeit.default_timer() - start,
                                                        {}))
                else:
                    log.warning(f'exchanges list is None!  prompt/response not saved')

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
