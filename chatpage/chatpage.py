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
    use_markdown: bool = False


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

                        # todo: latex: some models put "\[" and "\]" on sep lines, others just use the "$$" markers or [/] or ```latex, this is still far from perfect :(
                        latex_line = False
                        for line in rtext.results[ri].split('\n'):

                            if rtext.use_markdown:
                                # latex stuff
                                if line.strip() == '\\[':
                                    latex_line = True
                                    continue
                                if line.strip() == '\\]':
                                    latex_line = False
                                    continue
                                if latex_line or line.strip().startswith('[') and line.strip().endswith(']'):
                                    line = f'$${line}$$'

                                ui.markdown(content=line, extras=['fenced-code-blocks', 'tables', 'latex']).classes(result_text_classes)
                            else:
                                ui.label(line).classes(result_text_classes)

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

            # todo: increase encapsulation of InstanceData for some/all idata.<whatever> usages below
            idata.last_prompt_update()

            # loop the exchanges to build the texts needed to display
            responses: list[ResponseText] = []
            if idata.count_info_messages() == 0:
                for exchange in idata.chat_exchanges():
                    rtext: ResponseText = ResponseText(exchange.response_duration_secs, exchange.prompt)

                    # llm response
                    if exchange.llm_exchange is not None:
                        rtext.use_markdown = True
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
                            rtext.response_problems.append(f'exchange history (max:{idata.max_chat_exchanges()}) overflowed!  Oldest exchange dropped')

                    # vector store response
                    elif exchange.vector_store_response is not None:
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

            if idata.count_info_messages() > 0:
                rtext = ResponseText(response_duration_seconds=0.0, prompt=prompt)

                # info-messages are not exchanges/responses, e.g. they come from special commands
                for im in idata.info_messages():
                    rtext.results.append(im)
                idata.clear_info_messages()
                responses.append(rtext)

            # since this is NOT @refresh, we have to manually clear the scroll area
            scroller.clear()

            # display/render the various texts of the response
            render_response(responses, scroller)

            scroller.scroll_to(percent=1e6)  # 1e6 works around quasar scroll-area bug

        def do_llm(prompt: str, idata: InstanceData) -> LLMExchange:
            # todo: count tokens, etc.
            convo = [ex.llm_exchange for ex in idata.chat_exchanges() if ex.llm_exchange is not None]
            exchange: LLMExchange = idata.llm_config().chat_convo(convo=convo, prompt=prompt)
            return exchange

        async def handle_special_prompt(prompt: str, idata: InstanceData) -> None:
            log.info(f'(exchanges[{idata.chat_exchange_id()}]) prompt({idata.mode()}:{idata.llm_config().provider()}:{idata.llm_config().model_name}): "{prompt}"')

            # extract *n, e.g. "*2", "*3"...
            digit1: int = 0 if len(prompt) < 2 or (not prompt[1].isdigit()) else int(prompt[1])

            if len(prompt) == 1:
                idata.add_info_message(idata.special_about)
            elif prompt.startswith('*info'):
                idata.add_info_message('env:')
                for key in self.parms.keys():
                    val = self.parms[key]
                    if key.lower().endswith('_key') or key.lower().endswith('_token'):
                        val = config.redact(val)
                    idata.add_info_message(f'----{key}: {val}')
            elif prompt.startswith('*repeat'):
                if idata.mode_is_llm():
                    await handle_llm_prompt(idata.last_prompt(), idata)
                else:
                    await handle_vector_search_prompt(idata.last_prompt(), idata)
            elif prompt.startswith('*clear'):
                idata.clear_exchanges()
                idata.add_info_message('conversation cleared')
            # elif digit1 > 0:
            #     if 'n' in settings_selects:
            #         settings_selects['n'].set_value(digit1)
            #     await idata.change_n(digit1)
            else:
                idata.add_unknown_special_message(prompt)

        async def handle_llm_prompt(prompt: str, idata: InstanceData) -> None:
            # todo: suppress and note actually allowed parameters
            log.info(
                f'(exchanges[{idata.chat_exchange_id()}]) prompt({idata.mode()}:{idata.llm_config().provider()}:{idata.llm_config().model_name},'
                f'{idata.llm_config().settings().numbers_oneline_logging_str()}): "{prompt}"')
            # f'{idata.gllm_config().settings().temp},{idata.gllm_config().settings().top_p},{idata.gllm_config().settings().max_tokens}): "{prompt}"')

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
                                  source=idata.source(), mode=idata.mode())
                for choice_idx, sp_text in ce.problems().items():
                    log.warning(f'stop problem from prompt {prompt} choice[{choice_idx}]: {sp_text}')
                idata.add_chat_exchange(ce)

        async def handle_vector_search_prompt(prompt: str, idata: InstanceData) -> None:
            log.info(f'(exchanges[{idata.chat_exchange_id()}]) prompt({idata.mode()}:{idata.source()}): "{prompt}"')

            start = timeit.default_timer()

            vsresponse: VectorStoreResponse | None = None
            try:
                vsresponse = await run.io_bound(lambda: idata.vectorstore().search(prompt))
                log.debug(f'vector-search response: {vsresponse}')
            except (Exception,) as e:
                traceback.print_exc(file=sys.stdout)
                errmsg = f'vector-search error! {e.__class__.__name__}: {e}'
                log.warning(errmsg)
                ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)

            if vsresponse is not None:
                ce = ChatExchange(prompt, response_duration_secs=timeit.default_timer() - start, llm_exchange=None, vector_store_response=vsresponse,
                                  source=idata.source(), mode=idata.mode())
                idata.add_chat_exchange(ce)

        async def handle_enter(request, prompt_input: Input, spinner: Spinner, scroller: ScrollArea, idata: InstanceData) -> None:
            prompt_input.disable()
            prompt = prompt_input.value.strip()
            spinner.set_visibility(True)

            logstuff.update_from_request(request)  # updates logging prefix with info from each request

            if prompt_input.value.startswith('*'):
                await handle_special_prompt(prompt, idata)
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
                        pinput.on('keydown.enter', lambda req=request, i=idata: handle_enter(req, pinput, spinner, scroller, i))

                with (ui.column().classes('w-full flex-grow border-solid border border-white')):  # place-content-center')):
                    # the settings selection row
                    with (ui.row().classes('w-full border-solid border border-white')):  # place-content-center')):
                        sources = idata.all_sources()
                        ui.select(label='Source:',
                                  options=sources,
                                  value=idata.source(),
                                  ).on_value_change(lambda vc: call_and_focus(lambda: idata.change_source(vc.value), pinput, spinner)
                                                    ).tooltip('vs=vector search, llm=lang model chat').props('square outlined label-color=green').classes('min-w-30')

                        settings = idata.llm_config().settings() if idata.mode_is_llm() else idata.vectorstore().settings()
                        for sinfo in settings.info():
                            ui.select(label=sinfo.label,
                                      options=sinfo.options,
                                      value=sinfo.value,
                                      ).on_value_change(callback=lambda vc: call_and_focus(lambda: settings.change(vc.sender.props['label'], vc.value), pinput, spinner)
                                                        ).tooltip(sinfo.tooltip).props('square outlined label-color=green')

                    # the chat scroll area
                    with ui.scroll_area().classes('w-full flex-grow border border-solid border-white') as scroller:
                        await refresh_chat(pinput.value.strip(), idata, scroller)

            try:
                await ui.context.client.connected()
            except builtins.TimeoutError:
                log.warning(f'TimeoutError waiting for client connection, ignored')
            await pinput.run_method('focus')
