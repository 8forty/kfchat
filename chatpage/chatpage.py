import builtins
import logging
import sys
import timeit
import traceback
from dataclasses import dataclass, field

from fastapi import Request
from nicegui import ui, run, Client
from nicegui.elements.input import Input
from nicegui.elements.scroll_area import ScrollArea
from nicegui.elements.select import Select
from nicegui.elements.spinner import Spinner
from nicegui.events import Handler, ValueChangeEventArguments

import config
import frame
import logstuff
from chatexchanges import ChatExchange, VectorStoreResponse
from llmconfig.llmconfig import LLMConfig
from llmconfig.llmexchange import LLMExchange
from vectorstore.vsapi import VSAPI
from .instancedata import InstanceData

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


@dataclass
class ResponseText:
    response_duration_seconds: float
    prompt: str
    results: list[str] = field(default_factory=list)
    result_subscripts: list[list[str]] = field(default_factory=list)  # a list of e.g. metrics strings per result
    response_context = ''
    response_subscripts: list[str] = field(default_factory=list)
    response_problems: list[str] = field(default_factory=list)


class ChatPage:
    def __init__(self, all_llm_configs: dict[str, LLMConfig], init_llm_name: str, vectorstore: VSAPI, parms: dict[str, str]):
        # anything in here is shared by all instances of ChatPage
        self.all_llm_configs = all_llm_configs
        self.init_llm_name = init_llm_name
        self.vectorstore = vectorstore
        self.parms = parms

    def setup(self, path: str, pagename: str):

        def render_response(responses: list[ResponseText], scroller: ScrollArea):
            prompt_classes = 'w-full font-bold text-lg text-blue text-left px-2 pt-4 pb-1'
            result_text_classes = 'w-full text-lg text-white text-left px-2'
            subscript_classes = 'w-full italic text-xs text-slate-500 text-left px-10'
            problem_classes = 'w-full italic text-xs text-red text-left px-10'

            with scroller, ui.column().classes('w-full gap-y-0'):
                for rtext in responses:
                    # the prompt
                    ui.label(rtext.prompt).classes(prompt_classes)
                    ui.separator().props('size=4px')

                    # results
                    for ri in range(0, len(rtext.results)):
                        latex_line = False  # todo: some models put "\[" and "\]" on sep lines, others just use the "$$" markers or [/] or ```latex, this is still far from perfect :(
                        for line in rtext.results[ri].split('\n'):

                            if line.strip() == '\\[':
                                latex_line = True
                                continue
                            if line.strip() == '\\]':
                                latex_line = False
                                continue

                            if latex_line or line.strip().startswith('[') and line.strip().endswith(']'):
                                line = f'$${line}$$'
                            ui.markdown(content=line, extras=['fenced-code-blocks', 'tables', 'latex']).classes(result_text_classes)
                        # results-subscript
                        if len(rtext.result_subscripts) > ri:
                            for rinfo in rtext.result_subscripts[ri]:
                                ui.label(rinfo).classes(subscript_classes)

                    # response extra stuff
                    ui.label(f'[{rtext.response_context}]: {rtext.response_duration_seconds:.1f}s').classes(subscript_classes)
                    for ei in rtext.response_subscripts:
                        ui.label(f'{ei}').classes(subscript_classes)

                    # problems
                    for problem in rtext.response_problems:
                        ui.label(f'{problem}').classes(problem_classes)

        async def refresh_chat(prompt: str, idata: InstanceData, scroller: ScrollArea) -> None:
            # todo: local-storage-session to separate messages
            # todo: @refresh?

            # todo: increaase encapsulation of InstanceData for some/all idata.<whatever> usages below
            if idata.exchanges.len() > 0:
                idata.last_prompt = idata.exchanges.list()[-1].prompt

            # loop the exchanges to build the texts needed to display
            responses: list[ResponseText] = []

            if len(idata.info_messages) == 0:
                for exchange in idata.exchanges.list():
                    rtext: ResponseText = ResponseText(exchange.response_duration_secs, exchange.prompt)

                    # llm response
                    if exchange.llm_exchange is not None:
                        llm_exchange = exchange.llm_exchange
                        for response in llm_exchange.responses:
                            rtext.results.append(f'{response.content}')  # .classes(response_text_classes)
                            rtext.result_subscripts.append([f'logprobs: {"TBD"}'])
                        rtext.response_context += f'{exchange.mode},{llm_exchange.provider}:{llm_exchange.model_name},{llm_exchange.settings.numbers_oneline_logging_str()}'
                        rtext.response_subscripts.append(f'tokens:{llm_exchange.input_tokens}->{llm_exchange.output_tokens}')
                        rtext.response_subscripts.append(f'{llm_exchange.settings.texts_oneline_logging_str()}')

                        # stop problems
                        for choice_idx, stop_problem in exchange.problems().items():
                            rtext.response_problems.append(f'stop[{choice_idx}]:{stop_problem}')

                        # exchange problems
                        if exchange.overflowed():
                            rtext.response_problems.append(f'exchange history (max:{idata.exchanges.max_exchanges()}) overflowed!  Oldest exchange dropped')

                    # vector store response
                    elif exchange.vector_store_response is not None:
                        vs_resp = exchange.vector_store_response
                        rtext.response_context += f'{exchange.mode},{exchange.source}'
                        for result in exchange.vector_store_response.results:
                            rtext.results.append(f'[{exchange.mode}]: {result.content}')  # .classes(response_text_classes)

                            metric_list = []
                            for metric in result.metrics:
                                val = result.metrics[metric]
                                if val is not None and len(str(val)) > 0:
                                    if isinstance(val, float):
                                        metric_list.append(f'{metric}: {result.metrics[metric]:.03f}')
                                    else:
                                        metric_list.append(f'{metric}: {result.metrics[metric]}')
                            rtext.result_subscripts.append(metric_list)

                    responses.append(rtext)

            if len(idata.info_messages) > 0 or idata.unknown_special_message is not None:
                rtext = ResponseText(response_duration_seconds=0.0, prompt=prompt)
                if idata.unknown_special_message is not None:
                    rtext.results.append(idata.unknown_special_message)
                    idata.unknown_special_message = None

                # info-messages are not exchanges/responses, e.g. they come from special commands
                for im in idata.info_messages:
                    rtext.results.append(im)
                idata.info_messages.clear()
                responses.append(rtext)

            # since this is NOT @refresh, we have to manually clear the scroll area
            scroller.clear()
            # display/render the various texts of the response
            render_response(responses, scroller)

            scroller.scroll_to(percent=1e6)  # 1e6 works around quasar scroll-area bug

        def do_llm(prompt: str, idata: InstanceData) -> LLMExchange:
            # todo: count tokens, etc.
            convo = [ex.llm_exchange for ex in idata.exchanges.list() if ex.llm_exchange is not None]
            exchange: LLMExchange = idata.llm_config.chat_convo(convo=convo, prompt=prompt)
            return exchange

        async def handle_special_prompt(prompt: str, settings_selects: dict[str, Select], idata: InstanceData) -> None:
            log.info(f'(exchanges[{idata.exchanges.id()}]) prompt({idata.mode}:{idata.llm_config.provider()}:{idata.llm_config.model_name}): "{prompt}"')
            about = 'special commands: *, *info, *repeat, *clear, (n) *1/*2... '

            # extract *n, e.g. "*2", "*3"...
            digit1: int = 0 if len(prompt) < 2 or (not prompt[1].isdigit()) else int(prompt[1])

            if len(prompt) == 1:
                idata.info_messages.append(about)
            elif prompt.startswith('*info'):
                idata.info_messages.append('env:')
                for key in self.parms.keys():
                    val = self.parms[key]
                    if key.lower().endswith('_key') or key.lower().endswith('_token'):
                        val = config.redact(val)
                    idata.info_messages.append(f'----{key}: {val}')
            elif prompt.startswith('*repeat'):
                if idata.mode_is_llm():
                    await handle_llm_prompt(idata.last_prompt, idata)
                else:
                    await handle_vector_search_prompt(idata.last_prompt, idata)
            elif prompt.startswith('*clear'):
                idata.clear()
                idata.info_messages.append('conversation cleared')
            elif digit1 > 0:
                if 'n' in settings_selects:
                    settings_selects['n'].set_value(digit1)
                await idata.change_n(digit1)
            else:
                idata.unknown_special_message = f'{idata.unknown_special_prefix}: {prompt}; {about}'

        async def handle_llm_prompt(prompt: str, idata: InstanceData) -> None:
            log.info(
                f'(exchanges[{idata.exchanges.id()}]) prompt({idata.mode}:{idata.llm_config.provider()}:{idata.llm_config.model_name},'
                f'{idata.llm_config.settings().temp},{idata.llm_config.settings().top_p},{idata.llm_config.settings().max_tokens}): "{prompt}"')

            exchange: LLMExchange | None = None
            try:
                exchange = await run.io_bound(lambda: do_llm(prompt, idata))
            except (Exception,) as e:
                traceback.print_exc(file=sys.stdout)
                errmsg = f'llm error! {e.__class__.__name__}: {e}'
                log.warning(errmsg)
                ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)

            if exchange is not None:
                log.debug(f'llm exchange responses: {exchange.responses}')
                ce = ChatExchange(exchange.prompt, response_duration_secs=exchange.response_duration_secs,
                                  llm_exchange=exchange, vector_store_response=None,
                                  source=idata.source, mode=idata.mode)
                for choice_idx, sp_text in ce.problems().items():
                    log.warning(f'stop problem from prompt {prompt} choice[{choice_idx}]: {sp_text}')
                idata.exchanges.append(ce)

        async def handle_vector_search_prompt(prompt: str, idata: InstanceData) -> None:
            log.info(f'(exchanges[{idata.exchanges.id()}]) prompt({idata.mode}:{idata.source}): "{prompt}"')

            start = timeit.default_timer()

            vsresponse: VectorStoreResponse | None = None
            try:
                vsresponse = await run.io_bound(lambda: idata.vectorstore.search(prompt, howmany=idata.llm_config.settings().n))
                log.debug(f'vector-search response: {vsresponse}')
            except (Exception,) as e:
                traceback.print_exc(file=sys.stdout)
                errmsg = f'vector-search error! {e.__class__.__name__}: {e}'
                log.warning(errmsg)
                ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)

            if vsresponse is not None:
                ce = ChatExchange(prompt, response_duration_secs=timeit.default_timer() - start, llm_exchange=None, vector_store_response=vsresponse,
                                  source=idata.source, mode=idata.mode)
                idata.exchanges.append(ce)

        async def handle_enter(request, prompt_input: Input, spinner: Spinner, scroller: ScrollArea, settings_selects: dict[str, Select], idata: InstanceData) -> None:
            prompt_input.disable()
            prompt = prompt_input.value.strip()
            spinner.set_visibility(True)

            logstuff.update_from_request(request)  # updates logging prefix with info from each request

            if prompt_input.value.startswith('*'):
                await handle_special_prompt(prompt, settings_selects, idata)
            elif idata.mode_is_llm():
                await handle_llm_prompt(prompt, idata)
            else:
                await handle_vector_search_prompt(prompt, idata)

            spinner.set_visibility(False)
            prompt_input.value = ''
            prompt_input.enable()

            await refresh_chat(prompt, idata, scroller)
            await prompt_input.run_method('focus')

        async def call_and_focus(callback: Handler[ValueChangeEventArguments], prompt_input: Input, spinner: Spinner):
            prompt_input.disable()
            spinner.set_visibility(True)
            try:
                await callback()
            except (Exception,) as e:
                errmsg = f'change failed! {e.__class__.__name__}: {e}'
                log.warning(errmsg)
                traceback.print_exc(file=sys.stderr)
                ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)
            finally:
                prompt_input.enable()
                spinner.set_visibility(False)
                await prompt_input.run_method('focus')

        @ui.page(path=path)
        async def index(request: Request, client: Client) -> None:
            logstuff.update_from_request(request)
            log.info(f'route triggered')

            # this page sometimes take a bit of time to load (per: https://github.com/zauberzeug/nicegui/discussions/2429)
            try:
                # initial startup can be slow.....
                await client.connected(timeout=60.0)
            except builtins.TimeoutError:
                log.warning(f'TimeoutError waiting for client connection, ignored')

            idata = InstanceData(self.all_llm_configs, self.init_llm_name, self.vectorstore, self.parms)

            # setup the standard "frame" for all pages
            with frame.frame(f'{config.name} {pagename}'):

                # this suppresses the enormous default(?) 8px top/bottom margins on every line of markdown
                ui.add_css('div.nicegui-markdown p { margin-top: 0px; margin-bottom: 0px; }')

                # the footer is a "top-level" element in nicegui, so need not be setup in visual page order
                with ui.footer(bordered=True).classes('bg-black h-24'):
                    with ui.row().classes('w-full'):
                        spinner = ui.spinner(size='xl')
                        spinner.set_visibility(False)
                        pinput = ui.input(placeholder="Enter prompt").classes('flex-grow').props('rounded outlined color=primary bg-color=black')
                        pinput.on('keydown.enter', lambda req=request, i=idata: handle_enter(req, pinput, spinner, scroller, settings_selects, i))

                with (ui.column().classes('w-full flex-grow border-solid border border-white')):  # place-content-center')):
                    # the settings selection row
                    with (ui.row().classes('w-full border-solid border border-white')):  # place-content-center')):
                        sources = idata.all_sources()
                        selmodel = ui.select(label='Source:',
                                             options=sources,
                                             value=idata.source,
                                             ).on_value_change(lambda vc: call_and_focus(lambda: idata.change_source(vc.value), pinput, spinner)
                                                               ).tooltip('vs=vector search, llm=lang model chat').props('square outlined label-color=green').classes('min-w-30')

                        settings = idata.llm_config.settings
                        seln = ui.select(label='n:',
                                         options=[i for i in range(1, 10)],
                                         value=settings().n,
                                         ).on_value_change(lambda vc: call_and_focus(lambda: idata.change_n(vc.value), pinput, spinner)
                                                           ).tooltip('number of results per query').props('square outlined label-color=green').classes('min-w-20')
                        seltemp = ui.select(label='Temp:',
                                            options=[float(t) / 10.0 for t in range(0, 21)],
                                            value=settings().temp,
                                            ).on_value_change(lambda vc: call_and_focus(lambda: idata.change_temp(vc.value), pinput, spinner)
                                                              ).tooltip('responses: 0=very predictable, 2=very random/creative').props('square outlined label-color=green').classes('min-w-40')
                        seltopp = ui.select(label='Top_p:',
                                            options=[float(t) / 10.0 for t in range(0, 11)],
                                            value=settings().top_p,
                                            ).on_value_change(lambda vc: call_and_focus(lambda: idata.change_top_p(vc.value), pinput, spinner)
                                                              ).tooltip('responses: 0=less random, 1 more random').props('square outlined label-color=green').classes('min-w-40')
                        selmaxtok = ui.select(label='Max Tokens:',
                                              options=[80, 200, 400, 800, 1000, 1500, 2000],
                                              value=settings().max_tokens,
                                              ).on_value_change(lambda vc: call_and_focus(lambda: idata.change_max_tokens(vc.value), pinput, spinner)
                                                                ).tooltip('max tokens in response').props('square outlined label-color=green').classes('min-w-40')
                        sysmsg_names = [key for key in config.LLMData.sysmsg_all]
                        selsysmsg = ui.select(label='Sys Msg:',
                                              options=sysmsg_names,
                                              value=settings().system_message_name
                                              ).on_value_change(lambda vc: call_and_focus(lambda: idata.change_sysmsg(vc.value), pinput, spinner)
                                                                ).tooltip('system/setup text sent with each prompt').props('square outlined label-color=green').classes('min-w-50')

                        settings_selects = {'model': selmodel, 'n': seln, 'temp': seltemp, 'top_p': seltopp, 'maxtokens': selmaxtok, 'sysmsg': selsysmsg}

                    with ui.scroll_area().classes('w-full flex-grow border border-solid border-white') as scroller:
                        await refresh_chat(pinput.value.strip(), idata, scroller)

            try:
                await ui.context.client.connected()
            except builtins.TimeoutError:
                log.warning(f'TimeoutError waiting for client connection, ignored')
            await pinput.run_method('focus')
