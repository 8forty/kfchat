import builtins
import logging
import sys
import timeit
import traceback

from fastapi import Request
from nicegui import ui, run
from nicegui.elements.input import Input
from nicegui.elements.scroll_area import ScrollArea
from nicegui.elements.spinner import Spinner
from openai.types.chat import ChatCompletion

import config
import frame
import logstuff
from chatexchanges import ChatExchange, VectorStoreResponse, ChatExchanges
from llmconfig import LLMConfig
from vectorstorebase import VectorStoreBase

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class InstanceData:
    _next_id: int = 1

    def __init__(self, llm_config: LLMConfig, vectorstore: VectorStoreBase, env_values: dict[str, str]):
        self._id = InstanceData._next_id
        InstanceData._next_id += 1

        self.env_values: dict[str, str] = env_values
        self.exchanges: ChatExchanges = ChatExchanges(config.chat_exchanges_circular_list_count)

        # llm stuff
        self.llm_api_type: str = 'llm'
        self.llm_name_prefix: str = 'llm: '
        self.llm_config: LLMConfig = llm_config
        self.source_llm_name: str = f'{self.llm_name_prefix}{self.llm_config.model_name}'

        # vs stuff
        self.vs_api_type: str = 'vs'
        self.vs_name_prefix: str = 'vs: '
        self.vectorstore: VectorStoreBase = vectorstore

        # #### source info
        self.source_select_name: str = self.source_llm_name
        self.source_name: str = self.source_select_name  # name of the source object (we want to start with the llm, so select-name and name are the same)
        self.source_api: VectorStoreBase | None = None  # VS api or None for llm

    def api_type(self) -> str:
        return self.llm_api_type if self.source_api is None else self.vs_api_type

    def change_source(self, selected_name: str):
        if selected_name.startswith(self.llm_name_prefix):
            self.source_api = None
        else:
            self.source_api = self.vectorstore
            self.source_name = selected_name.removeprefix(self.vs_name_prefix)
        self.source_select_name = selected_name

    def source_names_list(self) -> list[str]:
        source_names: list[str] = [self.source_llm_name]
        source_names.extend([f'{self.vs_name_prefix}{name}' for name in self.vectorstore.list_collection_names()])
        source_names.sort(key=lambda k: 'zzz' + k if k.startswith(self.vs_name_prefix) else k)  # sort with the vs sources after the llm sources
        return source_names

    async def refresh_instance(self, scroller: ScrollArea) -> None:
        # todo: local-storage-session to separate messages
        scroller.clear()
        with scroller:
            if self.exchanges.len() > 0:
                response_text_classes = 'w-full font-bold text-lg text-green text-left px-10'
                response_subscript_classes = 'w-full italic text-xs text-black text-left px-10'

                for exchange in self.exchanges.list():

                    # the prompt
                    ui.label(exchange.prompt).classes('w-full font-bold text-lg text-blue text-left px-10')

                    # the response(s)
                    with (ui.column().classes('w-full gap-y-0')):
                        # subscript_context_info = f'{self.exchanges.id()},'
                        subscript_context_info = ''
                        subscript_results_info = ''
                        subscript_extra_info = ''
                        # todo: metrics, etc.
                        if exchange.llm_response is not None:
                            subscript_context_info += (f'{self.llm_api_type},{self.llm_config.model_api.api_type}:{self.llm_config.model_name}'
                                                       f',temp:{self.llm_config.temp},max_tokens:{self.llm_config.max_tokens}')
                            subscript_results_info += f'tokens:{exchange.llm_response.usage.prompt_tokens}/{exchange.llm_response.usage.completion_tokens}'
                            subscript_extra_info += f'{self.llm_config.system_message}'
                            for choice in exchange.llm_response.choices:
                                ui.label(f'[{self.llm_api_type}]: {choice.message.content}').classes(response_text_classes)

                        if exchange.vector_store_response is not None:
                            subscript_context_info += f'{self.vs_api_type},{self.source_name}'
                            for result in exchange.vector_store_response.results:
                                ui.label(f'[{self.vs_api_type}]: {result.content}').classes(response_text_classes)
                                ui.label(f'distance:{result.metrics['distance']:.03f}').classes(response_subscript_classes)

                        # subscripts
                        ui.label(f'[{subscript_context_info}]: '
                                 f'{subscript_results_info} '
                                 f'{exchange.response_duration_secs:.1f}s'
                                 ).classes(response_subscript_classes)
                        ui.label(f'{subscript_extra_info}').classes(response_subscript_classes)

                        # stop problems
                        stop_problems_string = ''
                        for choice_idx, stop_problem in exchange.stop_problems().items():
                            stop_problems_string += f'stop[{choice_idx}]:{stop_problem}'
                        if len(stop_problems_string) > 0:
                            ui.label(f'{stop_problems_string}').classes('w-full italic text-xs text-red text-left px-10')

                        # exchange problems
                        if exchange.overflowed():
                            ui.label(f'exchange history (max:{self.exchanges.max_exchanges()}) overflowed!  Oldest exchange dropped'
                                     ).classes('w-full italic text-xs text-red text-left px-10')
            else:
                ui.label('No messages yet').classes('mx-auto my-36 absolute-center text-2xl italic')

        scroller.scroll_to(percent=1.1)


class ChatPage:

    def __init__(self, llm_config: LLMConfig, vectorstore: VectorStoreBase, env_values: dict[str, str]):
        # anything in here is shared by all instances of ChatPage
        self.llm_config = llm_config
        self.vectorstore = vectorstore
        self.env_values = env_values

    def setup(self, path: str, pagename: str):

        def do_llm(prompt: str, idata: InstanceData) -> ChatCompletion | None:
            # todo: count tokens, etc.
            completion = idata.llm_config.model_api.llm_run_prompt(idata.llm_config.model_name,
                                                                   temp=idata.llm_config.temp,
                                                                   max_tokens=idata.llm_config.max_tokens,
                                                                   n=1,  # todo: openai:any value works, ollama: 1 resp for any value, groq: only 1 allowed
                                                                   sysmsg=idata.llm_config.system_message,
                                                                   prompt=prompt,
                                                                   convo=idata.exchanges)
            return completion

        def do_vector_search(prompt: str, idata: InstanceData):
            vsresponse = idata.source_api.ask(prompt, collection_name=idata.source_name)
            return vsresponse

        async def handle_enter_llm(request, prompt_input: Input, spinner: Spinner, scroller: ScrollArea, idata: InstanceData) -> None:
            prompt = prompt_input.value.strip()
            log.info(
                f'(exchanges[{idata.exchanges.id()}]) prompt({idata.api_type()}:{idata.llm_config.model_api.api_type}:{idata.llm_config.model_name},{idata.llm_config.temp},{idata.llm_config.max_tokens}): "{prompt}"')
            prompt_input.disable()
            logstuff.update_from_request(request)  # updates logging prefix with info from each request

            start = timeit.default_timer()
            spinner.set_visibility(True)

            # todo: file of prompts
            # if prompt.startswith('*'):  # load a file of prompts
            #     with

            llm_response: ChatCompletion | None = None
            try:
                llm_response = await run.io_bound(do_llm, prompt, idata)
            except (Exception,) as e:
                traceback.print_exc(file=sys.stdout)
                log.warning(f'llm error! {e}')
                ui.notify(message=f'llm error! {e}', position='top', type='negative', close_button='Dismiss', timeout=0)

            spinner.set_visibility(False)

            if llm_response is not None:
                log.debug(f'llm response: {llm_response}')
                ce = ChatExchange(prompt, response_duration_secs=timeit.default_timer() - start,
                                  llm_response=llm_response, vector_store_response=None)
                for choice_idx, sp_text in ce.stop_problems().items():
                    log.warning(f'stop problem from prompt {prompt} choice[{choice_idx}]: {sp_text}')
                idata.exchanges.append(ce)

            prompt_input.value = ''
            prompt_input.enable()
            await idata.refresh_instance(scroller)
            scroller.scroll_to(percent=100.0, axis='vertical', duration=0.0)
            await prompt_input.run_method('focus')

            # make sure client is connected before auto-scroll
            # try:
            #     await ui.context.client.connected()
            # except builtins.TimeoutError:
            #     pass
            # scroller.scroll_to(percent=1.0, axis='vertical', duration=0.0)

        async def handle_enter_vector_search(request, prompt_input: Input, spinner: Spinner, scroller: ScrollArea, idata: InstanceData) -> None:
            prompt = prompt_input.value.strip()
            log.info(f'(exchanges[{idata.exchanges.id()}]) prompt({idata.api_type()}:{idata.source_name}): "{prompt}"')
            prompt_input.disable()
            logstuff.update_from_request(request)  # updates logging prefix with info from each request

            start = timeit.default_timer()
            spinner.set_visibility(True)

            # todo: file of prompts
            # if prompt.startswith('*'):  # load a file of prompts
            #     with

            vsresponse: VectorStoreResponse | None = None
            try:
                vsresponse = await run.io_bound(do_vector_search, prompt, idata)
                log.debug(f'vector-search response: {vsresponse}')
            except (Exception,) as e:
                traceback.print_exc(file=sys.stdout)
                log.warning(f'vector-search error! {e}')
                ui.notify(message=f'vector-search error! {e}', position='top', type='negative', close_button='Dismiss', timeout=0)

            spinner.set_visibility(False)

            if vsresponse is not None:
                ce = ChatExchange(prompt, response_duration_secs=timeit.default_timer() - start, llm_response=None, vector_store_response=vsresponse)
                idata.exchanges.append(ce)

            prompt_input.value = ''
            prompt_input.enable()
            await idata.refresh_instance(scroller)
            # scroller.scroll_to(percent=100.0, axis='vertical', duration=0.0)
            await prompt_input.run_method('focus')

            # make sure client is connected before auto-scroll
            try:
                await ui.context.client.connected()
            except builtins.TimeoutError:
                pass
            scroller.scroll_to(percent=1.0, axis='vertical', duration=0.0)

        async def handle_enter(request, prompt_input: Input, spinner: Spinner, scroller: ScrollArea, idata: InstanceData) -> None:
            if idata.source_api is None:
                await handle_enter_llm(request, prompt_input, spinner, scroller, idata)
            else:
                await handle_enter_vector_search(request, prompt_input, spinner, scroller, idata)

        @ui.page(path)
        async def index(request: Request) -> None:
            logstuff.update_from_request(request)
            log.info(f'route triggered')

            idata = InstanceData(self.llm_config, self.vectorstore, self.env_values)

            # setup the standard "frame" for all pages
            with frame.frame(f'{config.name} {pagename}', 'bg-white'):
                with (ui.column().classes('w-full flex-grow border-solid border border-black')):  # place-content-center')):
                    # the source selection/info row
                    with (ui.row().classes('w-full border-solid border border-black')):  # place-content-center')):
                        source_names = idata.source_names_list()
                        ui.select(label='Source:',
                                  options=source_names,
                                  value=idata.source_select_name,
                                  ).on_value_change(lambda vc: idata.change_source(vc.value)).props('square outlined label-color=green')

                    # with ui.scroll_area(on_scroll=lambda e: print(f'~~~~ e: {e}')).classes('w-full flex-grow border border-solid border-black') as scroller:
                    with ui.scroll_area().classes('w-full flex-grow border border-solid border-black') as scroller:
                        await idata.refresh_instance(scroller)

            # the footer is a "top-level" element in nicegui, so need not be setup in visual page order
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
                    prompt_input.on('keydown.enter', lambda req=request, i=idata: handle_enter(req, prompt_input, spinner, scroller, i))

            try:
                await ui.context.client.connected()
            except builtins.TimeoutError:
                log.warning(f'TimeoutError waiting for client connection, connection ignored')
            await prompt_input.run_method('focus')
