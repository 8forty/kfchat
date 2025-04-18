import builtins
import logging
import sys
import timeit
import traceback
from dataclasses import dataclass, field

from fastapi import Request
from nicegui import ui, run, Client
from nicegui.elements.input import Input
from nicegui.elements.label import Label
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
    use_markdown: bool = False


class ChatPage:
    def __init__(self, all_llm_configs: dict[str, LLMConfig], init_llm_name: str, vectorstore: VSAPI, parms: dict[str, str]):
        # anything in here is shared by all instances of ChatPage
        self.all_llm_configs = all_llm_configs
        self.init_llm_name = init_llm_name
        self.vectorstore = vectorstore
        self.parms = parms

    def setup(self, path: str, pagename: str):
        settings_selects: dict[str, Select] = {}

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
                        is_latex_line = False
                        for line in rtext.results[ri].split('\n'):
                            raw_line = line

                            if rtext.use_markdown:
                                # dollar signs tend to get crazy with markdown, after several tries from the stack overflow below, this one finally worked: '&#36;'
                                # https://stackoverflow.com/questions/16089089/escaping-dollar-sign-in-ipython-notebook
                                line = line.replace('$', '&#36;')

                                # latex stuff
                                if line.strip() == '\\[':
                                    is_latex_line = True
                                    continue
                                if line.strip() == '\\]':
                                    is_latex_line = False
                                    continue
                                if is_latex_line or line.strip().startswith('[') and line.strip().endswith(']'):
                                    log.debug('latex line in results')
                                    line = f'$${line}$$'

                                if raw_line != line:
                                    log.debug(f'modified line! {line}')
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
                            rtext.results.append(f'{result.content}')

                            metric_list: list[str] = [f'id: {result.result_id}']
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
            convo = [ex.llm_exchange for ex in idata.chat_exchanges() if ex.llm_exchange is not None]
            exchange: LLMExchange = idata.llm_config().chat_convo(convo=convo, prompt=prompt)
            return exchange

        async def handle_special_prompt(prompt: str, idata: InstanceData) -> None:
            log.info(f'(exchanges[{idata.chat_exchange_id()}]) prompt({idata.mode()}:{idata.llm_config().provider()}:{idata.llm_config().model_name}): "{prompt}"')

            # extract *n, e.g. "*2", "*3"...
            digit1: int = 0 if len(prompt) < 2 or (not prompt[1].isdigit()) else int(prompt[1])

            if len(prompt) == 1:
                idata.add_info_message(idata.special_about())
            elif prompt.startswith('*info'):
                idata.add_info_message('env:')
                for key in self.parms.keys():
                    val = self.parms[key]
                    if key.lower().endswith('_key') or key.lower().endswith('_token'):
                        val = config.redact(val)
                    idata.add_info_message(f'----{key}: {val}')
            elif prompt.startswith('*repeat'):
                if idata.last_prompt() is not None:
                    if idata.mode_is_llm():
                        await handle_llm_prompt(idata.last_prompt(), idata)
                    else:
                        await handle_vector_search_prompt(idata.last_prompt(), idata)
                else:
                    idata.add_info_message('no previous prompt!')
            elif prompt.startswith('*clear'):
                idata.clear_exchanges()
                idata.add_info_message('conversation cleared')
            elif digit1 > 0:
                if idata.mode_is_llm():
                    settings = idata.llm_config().settings()
                    # only change n if the source has an n spec
                    for spec in settings.specs():
                        if spec.label == 'n':
                            await settings.change(spec.label, digit1)
                            # now update the gui
                            if 'n' in settings_selects.keys():
                                settings_selects['n'].value = digit1
            else:
                idata.add_unknown_special_message(prompt)

        async def run_llm_prompt(prompt: str, idata: InstanceData) -> LLMExchange | None:
            # todo: suppress + note actually allowed parameters
            log.info(
                f'(exchanges[{idata.chat_exchange_id()}]) prompt({idata.mode()}:{idata.llm_config().provider()}:{idata.llm_config().model_name},'
                f'{idata.llm_config().settings().numbers_oneline_logging_str()}): "{prompt}"')

            exchange: LLMExchange | None = None
            try:
                exchange = await run.io_bound(lambda: do_llm(prompt, idata))
            except (Exception,) as e:
                traceback.print_exc(file=sys.stdout)
                errmsg = f'llm error! {e.__class__.__name__}: {e}'
                log.warning(errmsg)
                ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)

            return exchange

        async def handle_llm_prompt(prompt: str, idata: InstanceData) -> None:
            exchange: LLMExchange | None = await run_llm_prompt(prompt, idata)
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
                # todo: configure dense_weight; stop using llm_config!
                vsresponse = await run.io_bound(lambda: idata.vectorstore().hybrid_search(query=prompt, max_results=idata.llm_config().settings().value('n'), dense_weight=0.5))
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

        async def handle_rag_prompt(prompt: str, idata: InstanceData) -> None:
            log.info(f'(exchanges[{idata.chat_exchange_id()}]) prompt({idata.mode()}:{idata.source()}): "{prompt}"')

            # first get the vector results
            try:
                # todo: configure max_results and dense_weight
                vsresponse: VectorStoreResponse = await run.io_bound(lambda: idata.vectorstore().hybrid_search(query=prompt, max_results=0, dense_weight=0.5))
                log.debug(f'rag vector-search response: {vsresponse}')
                context = [r.content for r in vsresponse.results]
                if len(context) > 0:
                    exchange: LLMExchange | None = await run_llm_prompt(config.LLMData.rag1_prompt.format(context=context, query=prompt), idata)
                    if exchange is not None:
                        log.debug(f'rag llm exchange responses: {exchange.responses}')
                        ce = ChatExchange(prompt, response_duration_secs=exchange.response_duration_secs,
                                          llm_exchange=exchange, vector_store_response=None,
                                          source=idata.source(), mode=idata.mode())
                        for choice_idx, sp_text in ce.problems().items():
                            log.warning(f'rag stop problem from prompt {prompt} choice[{choice_idx}]: {sp_text}')
                        idata.add_chat_exchange(ce)
                    else:
                        raise ValueError(f'rag got no result from LLM! ({idata.source()})')
                else:
                    raise ValueError(f'rag found no context! ({idata.source()})')

            except (Exception,) as e:
                traceback.print_exc(file=sys.stdout)
                errmsg = f'rag error! {e.__class__.__name__}: {e}'
                log.warning(errmsg)
                ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)

        async def handle_enter(request, prompt_input: Input, spinner: Spinner, scroller: ScrollArea, idata: InstanceData) -> None:
            prompt_input.disable()
            prompt = prompt_input.value.strip()
            spinner.set_visibility(True)

            logstuff.update_from_request(request)  # updates logging prefix with info from each request

            if prompt_input.value.startswith('*'):
                await handle_special_prompt(prompt, idata)
            elif idata.mode_is_llm():
                await handle_llm_prompt(prompt, idata)
            elif idata.mode_is_vs():
                await handle_vector_search_prompt(prompt, idata)
            elif idata.mode_is_rag():
                await handle_rag_prompt(prompt, idata)
            else:
                raise ValueError(f'unknown mode: {idata.mode()}')

            spinner.set_visibility(False)
            prompt_input.value = ''
            prompt_input.enable()

            await refresh_chat(prompt, idata, scroller)
            await prompt_input.run_method('focus')

        async def change_source(source_label: Label, new_value: str, callback: Handler[ValueChangeEventArguments], prompt_input: Input, spinner: Spinner):
            source_label.set_text(new_value)
            await call_and_focus(callback, prompt_input, spinner)

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

            # creating instance data here (instead of during setup) ensures separate windows/tabs have separate data
            idata = InstanceData(self.all_llm_configs, self.init_llm_name, self.vectorstore, self.parms)

            # setup the standard "frame" for all pages
            with (frame.frame(f'{config.name} {pagename}')):

                # this suppresses the enormous default(?) 8px top/bottom margins on every line of markdown
                ui.add_css('div.nicegui-markdown p { margin-top: 0px; margin-bottom: 0px; }')

                # the footer is a "top-level" element in nicegui, so need not be setup in visual page order
                with ui.footer(bordered=True).classes('bg-black h-24'):
                    with ui.row().classes('w-full'):
                        spinner = ui.spinner(size='xl')
                        spinner.set_visibility(False)
                        pinput = ui.input(placeholder="Enter prompt").classes('flex-grow').props('rounded outlined color=primary bg-color=black')
                        pinput.on('keydown.enter', lambda req=request, i=idata: handle_enter(req, pinput, spinner, scroller, i))

                with (ui.column().classes('w-full flex-grow border-solid border border-white')):
                    # the settings selection row
                    settings = idata.llm_config().settings() if idata.mode_is_llm() else idata.vectorstore().settings()
                    with ui.row().classes(f'w-full border-solid border border-white grid grid-cols-{len(settings.specs()) + 2} gap-0'):
                        sources = idata.all_sources()
                        with ui.column().classes('border grid grid-rows-2 gap-0 col-span-2'):
                            ui.label('source').classes('text-green text-[12px] font-[400] m-1')
                            with ui.label(text=idata.llm_source(idata.llm_config())
                                          ).classes('w-full m-1').tooltip('vs=vector search, llm=lang model chat, rag=rag') as source_label:
                                with ui.menu():
                                    for k, v in sources.items():
                                        with ui.menu_item(k, auto_close=False):
                                            with ui.item_section().props('side'):
                                                ui.icon('keyboard_arrow_right')
                                            with ui.menu().props('anchor="top end" self="top start" auto-close'):
                                                for sub_k in v:
                                                    ui.menu_item(sub_k, on_click=lambda ceargs: change_source(source_label,
                                                                                                              ceargs.sender.default_slot.children[0].text,
                                                                                                              lambda: idata.change_source(ceargs.sender.default_slot.children[0].text),
                                                                                                              pinput,
                                                                                                              spinner))
                            # source_label = ui.label(idata.llm_source(idata.llm_config())).classes('w-full')

                        for sinfo in settings.specs():
                            # vc.sender.props['label'] is 'n', 'temp', ...
                            s = ui.select(label=sinfo.label,
                                          options=sinfo.options,
                                          value=sinfo.value,
                                          ).on_value_change(callback=lambda vc: call_and_focus(lambda: settings.change(vc.sender.props['label'], vc.value), pinput, spinner)
                                                            ).tooltip(sinfo.tooltip).props('square outlined label-color=green')
                            settings_selects[sinfo.label] = s

                            # the chat scroll area
                    with ui.scroll_area().classes('w-full flex-grow border border-solid border-white') as scroller:
                        await refresh_chat(pinput.value.strip(), idata, scroller)

            try:
                await ui.context.client.connected()
            except builtins.TimeoutError:
                log.warning(f'TimeoutError waiting for client connection, ignored')
            await pinput.run_method('focus')
