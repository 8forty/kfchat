import builtins
import logging
import sys
import timeit
import traceback

from fastapi import Request
from nicegui import ui, run, Client
from nicegui.element import Element
from nicegui.elements.input import Input
from nicegui.elements.scroll_area import ScrollArea
from nicegui.elements.spinner import Spinner
from nicegui.events import Handler, ValueChangeEventArguments

import config
import data
import frame
import logstuff
from chatexchanges import ChatExchange, VectorStoreResponse, LLMOaiResponse
from llmoaiconfig import LLMOaiExchange, LLMOaiConfig
from vsapi import VSAPI
from .instancedata import InstanceData

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class ChatPage:

    def __init__(self, llm_configs: dict[str, LLMOaiConfig], init_llm_model_name: str, vectorstore: VSAPI, parms: dict[str, str]):
        # anything in here is shared by all instances of ChatPage
        self.llm_configs = llm_configs
        self.llm_config = llm_configs[init_llm_model_name]
        self.vectorstore = vectorstore
        self.parms = parms

    def setup(self, path: str, pagename: str):

        def do_llm(prompt: str, idata: InstanceData) -> LLMOaiExchange:
            # todo: count tokens, etc.
            convo = [LLMOaiExchange(ex.prompt, ex.llm_response.chat_completion) for ex in idata.exchanges.list() if ex.llm_response is not None]
            exchange: LLMOaiExchange = idata.llm_config.chat_convo(convo=convo, prompt=prompt)
            return exchange

        async def handle_enter_special(request, prompt_input: Input, spinner: Spinner, scroller: ScrollArea, idata: InstanceData) -> None:
            logstuff.update_from_request(request)  # updates logging prefix with info from each request
            prompt = prompt_input.value.strip()
            log.info(f'(exchanges[{idata.exchanges.id()}]) prompt({idata.current_source_type}:{idata.llm_config.api_type()}:{idata.llm_config.model_name}): "{prompt}"')

            prompt_input.disable()
            about = 'special commands: *, *info, *repeat, *forget'

            if len(prompt) == 1:
                idata.info_messages.append(about)
            elif prompt.startswith('*info'):
                idata.info_messages.append('env:')
                for key in self.parms.keys():
                    idata.info_messages.append(f'----{key}: {self.parms[key]}')
            elif prompt.startswith('*repeat'):
                prompt_input.set_value(idata.last_prompt)
                await handle_enter_llm(request, prompt_input, spinner, scroller, idata)
            elif prompt.startswith('*forget'):
                idata.forget()
                idata.info_messages.append('conversation forgotten')
            else:
                idata.info_messages.append(f'unknown special command: {prompt}; {about}')

            spinner.set_visibility(False)

            prompt_input.value = ''
            prompt_input.enable()
            await idata.refresh_instance(scroller)
            await prompt_input.run_method('focus')

        async def handle_enter_llm(request, prompt_input: Input, spinner: Spinner, scroller: ScrollArea, idata: InstanceData) -> None:
            logstuff.update_from_request(request)  # updates logging prefix with info from each request
            prompt = prompt_input.value.strip()
            log.info(
                f'(exchanges[{idata.exchanges.id()}]) prompt({idata.current_source_type}:{idata.llm_config.api_type()}:{idata.llm_config.model_name},'
                f'{idata.llm_config.settings.temp},{idata.llm_config.settings.top_p},{idata.llm_config.settings.max_tokens}): "{prompt}"')
            prompt_input.disable()

            start = timeit.default_timer()
            spinner.set_visibility(True)

            exchange: LLMOaiExchange | None = None
            try:
                exchange = await run.io_bound(lambda: do_llm(prompt, idata))
            except (Exception,) as e:
                traceback.print_exc(file=sys.stdout)
                log.warning(f'llm error! {e}')
                ui.notify(message=f'llm error! {e}', position='top', type='negative', close_button='Dismiss', timeout=0)

            spinner.set_visibility(False)

            if exchange is not None:
                log.debug(f'chat completion: {exchange.completion}')
                ce = ChatExchange(exchange.prompt, response_duration_secs=timeit.default_timer() - start,
                                  llm_response=LLMOaiResponse(exchange.completion, idata.llm_config), vector_store_response=None)
                for choice_idx, sp_text in ce.stop_problems().items():
                    log.warning(f'stop problem from prompt {prompt} choice[{choice_idx}]: {sp_text}')
                idata.exchanges.append(ce)

            prompt_input.value = ''
            prompt_input.enable()
            await idata.refresh_instance(scroller)
            await prompt_input.run_method('focus')

        async def handle_enter_vector_search(request, prompt_input: Input, spinner: Spinner, scroller: ScrollArea, idata: InstanceData) -> None:
            logstuff.update_from_request(request)  # updates logging prefix with info from each request
            prompt = prompt_input.value.strip()
            log.info(f'(exchanges[{idata.exchanges.id()}]) prompt({idata.current_source_type}:{idata.source_name}): "{prompt}"')
            prompt_input.disable()

            start = timeit.default_timer()
            spinner.set_visibility(True)

            vsresponse: VectorStoreResponse | None = None
            try:
                # vsresponse = await run.io_bound(lambda: do_vector_search(prompt, idata))
                vsresponse = await run.io_bound(lambda: idata.vectorstore.search(prompt, howmany=idata.llm_config.settings.n))
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
            await prompt_input.run_method('focus')

        async def handle_enter(request, prompt_input: Input, spinner: Spinner, scroller: ScrollArea, idata: InstanceData) -> None:
            if prompt_input.value.startswith('*'):
                await handle_enter_special(request, prompt_input, spinner, scroller, idata)
            elif idata.source_type_is_llm():
                await handle_enter_llm(request, prompt_input, spinner, scroller, idata)
            else:
                await handle_enter_vector_search(request, prompt_input, spinner, scroller, idata)

        async def change_and_focus(callback: Handler[ValueChangeEventArguments], focus_element: Element):
            await callback()
            await focus_element.run_method('focus')

        @ui.page(path=path)
        async def index(request: Request, client: Client) -> None:
            logstuff.update_from_request(request)
            log.info(f'route triggered')

            # this page sometimes take a bit of time to load (per: https://github.com/zauberzeug/nicegui/discussions/2429)
            await client.connected(timeout=3.0)

            idata = InstanceData(self.llm_configs, self.llm_config, self.vectorstore, self.parms)

            # setup the standard "frame" for all pages
            with frame.frame(f'{config.name} {pagename}', 'bg-white'):
                with (ui.column().classes('w-full flex-grow border-solid border border-black')):  # place-content-center')):
                    # the source selection/info row
                    with (ui.row().classes('w-full border-solid border border-black')):  # place-content-center')):
                        source_names = idata.source_names_list()
                        settings = self.llm_config.settings
                        ui.select(label='Source:',
                                  options=source_names,
                                  value=idata.source_select_name,
                                  ).on_value_change(lambda vc: change_and_focus(lambda: idata.change_source(vc.value, spinner, pinput), pinput)).props('square outlined label-color=green')
                        ui.select(label='n:',
                                  options=[i for i in range(1, 10)],
                                  value=settings.n,
                                  ).on_value_change(lambda vc: change_and_focus(lambda: idata.change_n(vc.value), pinput)).props('square outlined label-color=green')
                        ui.select(label='Temp:',
                                  options=[float(t) / 10.0 for t in range(0, 21)],
                                  value=settings.temp,
                                  ).on_value_change(lambda vc: change_and_focus(lambda: idata.change_temp(vc.value), pinput)).props('square outlined label-color=green')
                        ui.select(label='Top_p:',
                                  options=[float(t) / 10.0 for t in range(0, 11)],
                                  value=settings.top_p,
                                  ).on_value_change(lambda vc: change_and_focus(lambda: idata.change_top_p(vc.value), pinput)).props('square outlined label-color=green')
                        ui.select(label='Max Tokens:',
                                  options=[80, 200, 400, 800, 1000, 1500, 2000],
                                  value=settings.max_tokens,
                                  ).on_value_change(lambda vc: change_and_focus(lambda: idata.change_max_tokens(vc.value), pinput)).props('square outlined label-color=green')
                        sysmsg_names = [key for key in data.sysmsg_all]
                        ui.select(label='Sys Msg:',
                                  options=sysmsg_names,
                                  value=settings.system_message_name
                                  ).on_value_change(lambda vc: change_and_focus(lambda: idata.change_sysmsg(vc.value), pinput)).props('square outlined label-color=green')

                    # with ui.scroll_area(on_scroll=lambda e: print(f'~~~~ e: {e}')).classes('w-full flex-grow border border-solid border-black') as scroller:
                    with ui.scroll_area().classes('w-full flex-grow border border-solid border-black') as scroller:
                        await idata.refresh_instance(scroller)

            # the footer is a "top-level" element in nicegui, so need not be setup in visual page order
            with ui.footer().classes('bg-slate-100 h-24'):
                with ui.row().classes('w-full'):
                    spinner = ui.spinner(size='xl')
                    spinner.set_visibility(False)
                    pinput = ui.input(placeholder="Enter prompt").classes('flex-grow').props('rounded outlined').props('color=primary').props('bg-color=white')
                    pinput.on('keydown.enter', lambda req=request, i=idata: handle_enter(req, pinput, spinner, scroller, i))

            try:
                await ui.context.client.connected()
            except builtins.TimeoutError:
                log.warning(f'TimeoutError waiting for client connection, ignored')
            await pinput.run_method('focus')
