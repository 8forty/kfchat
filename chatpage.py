import builtins
import logging
import sys
import timeit
import traceback

import openai
from fastapi import Request
from nicegui import ui, run
from nicegui.elements.input import Input
from nicegui.elements.spinner import Spinner
from openai.types.chat import ChatCompletion

import llmopenaiapi
import config
import frame
import logstuff
import vectorstore_chroma
from chatexchanges import ChatExchange, VectorStoreResponse
from modelapi import ModelAPI
from vectorstore_chroma import VectorStoreChroma

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class LLMConfig:
    def __init__(self, api_type: str, env_values: dict[str, str]):
        # todo: these should come from e.g. pref screen
        self.model_api: ModelAPI = ModelAPI(api_type, parms=env_values)
        self.model_name: str = ['llama3.2:1b', 'llama3.2:3b', 'llama3.3:70b', 'qwen2.5:0.5b', 'gemma2:2b', 'qwq'][1]
        self.temp: float = 0.7
        self.max_tokens: int = 80
        self.system_message: str = (f'You are a helpful chatbot that talks in a conversational manner. '
                                    f'Your responses must always be less than {self.max_tokens} tokens.')
        self.client: openai.OpenAI = self.model_api.client()


class InstanceData:

    def __init__(self, llm_config: LLMConfig, env_values: dict[str, str]):
        self.llm_name_prefix: str = 'llm: '
        self.vs_name_prefix: str = 'VS: '
        self.llm_config: LLMConfig = llm_config
        self.env_values: dict[str, str] = env_values
        self.exchanges: llmopenaiapi.ChatExchanges = llmopenaiapi.ChatExchanges(config.chat_exchanges_circular_list_count)
        self.llm = llmopenaiapi.LLMOpenaiAPI(llm_config.client)

        # #### source info
        self.source_llm_name: str = f'{self.llm_name_prefix}{self.llm_config.model_name}'
        self.source_select_name: str = self.source_llm_name
        self.source_name: str = self.source_select_name  # name of the source object (we want to start with the llm, so select-name and name are the same)
        self.source_api: VectorStoreChroma | None = None  # VS api or None for llm

    def api_type(self) -> str:
        return 'llm' if self.source_api is None else 'vs'

    def change_source(self, selected_name: str):
        if selected_name.startswith(self.llm_name_prefix):
            self.source_api = None
        else:
            self.source_api = VectorStoreChroma(vectorstore_chroma.chromadb_client, self.env_values)
            self.source_name = selected_name.removeprefix(self.vs_name_prefix)
        self.source_select_name = selected_name

    @ui.refreshable
    async def refresh_chat_exchanges(self, llm_config: LLMConfig) -> None:
        vectorstore_chroma.setup_once(self.env_values)

        # the chat source selection/info row
        with (ui.row().classes('w-full border-solid border border-black')):  # place-content-center')):
            source_names: list[str] = [self.source_llm_name]
            source_names.extend([f'{self.vs_name_prefix}{c.name}' for c in vectorstore_chroma.chromadb_client.list_collections()])
            source_names.sort(key=lambda k: 'zzz' + k if k.startswith(self.vs_name_prefix) else k)
            ui.select(label='Source:',
                      options=source_names,
                      value=self.source_select_name,
                      ).on_value_change(lambda vc: self.change_source(vc.value)).props('square outlined label-color=green')

        # todo: local-storage-session to separate messages
        if self.exchanges.len() > 0:
            for exchange in self.exchanges.list():

                # the prompt
                ui.label(exchange.prompt).classes('w-full font-bold text-lg text-blue text-left px-10')

                with ui.column().classes('w-full gap-y-0'):
                    # the response(s)
                    context_info = f'{self.exchanges.id()}'
                    completion_extra = ''
                    # todo: metrics, etc.
                    if exchange.llm_response is not None:
                        context_info += f',{self.llm_config.model_api.api_type}:{llm_config.model_name},{llm_config.temp},{llm_config.max_tokens}'
                        completion_extra = f'{exchange.llm_response.usage.prompt_tokens}/{exchange.llm_response.usage.completion_tokens} '
                        for choice in exchange.llm_response.choices:
                            ui.label(f'[llm]: {choice.message.content}').classes('w-full font-bold text-lg text-green text-left px-10')

                    if exchange.vector_store_response is not None:
                        for result in exchange.vector_store_response.results:
                            ui.label(f'[vs]: {result.content}').classes('w-full font-bold text-lg text-green text-left px-10')
                            ui.label(f'distance: {result.metrics['distance']:.03f}').classes('w-full italic text-xs text-black text-left px-10')

                    # the context info
                    ui.label(f'[{context_info}]: '
                             f'{completion_extra}'
                             f'{exchange.response_duration_secs:.1f}s'
                             ).classes('w-full italic text-xs text-black text-left px-10')
                    if exchange.llm_response is not None:
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

    def __init__(self, env_values: dict[str, str]):
        # anything in here is shared by all instances of ChatPage
        self.env_values = env_values

        # todo: configure this
        self.llm_config = LLMConfig('ollama', self.env_values)

    def setup(self, path: str, pagename: str):

        def do_llm(prompt: str, idata: InstanceData) -> ChatCompletion | None:
            # todo: count tokens, etc.
            completion = idata.llm.llm_run_prompt(self.llm_config.model_name,
                                                  temp=self.llm_config.temp, max_tokens=self.llm_config.max_tokens,
                                                  n=2,  # todo: this doesn't work for ?? ollama:??
                                                  sysmsg=self.llm_config.system_message,
                                                  prompt=prompt,
                                                  convo=idata.exchanges)
            return completion

        def do_vector_search(prompt: str, idata: InstanceData):
            vsresponse = idata.source_api.ask(prompt, collection_name=idata.source_name)
            return vsresponse

        async def handle_enter_llm(request, prompt_input: Input, spinner: Spinner, idata: InstanceData) -> None:
            prompt = prompt_input.value.strip()
            log.info(f'(exchanges[{idata.exchanges.id()}]) prompt({idata.api_type()}:{self.llm_config.model_api.api_type}:{self.llm_config.model_name},{self.llm_config.temp},{self.llm_config.max_tokens}): "{prompt}"')
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
            idata.refresh_chat_exchanges.refresh(self.llm_config)
            await prompt_input.run_method('focus')

        async def handle_enter_vector_search(request, prompt_input: Input, spinner: Spinner, idata: InstanceData) -> None:
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
            idata.refresh_chat_exchanges.refresh(self.llm_config)
            await prompt_input.run_method('focus')

        async def handle_enter(request, prompt_input: Input, spinner: Spinner, idata: InstanceData) -> None:
            if idata.source_api is None:
                await handle_enter_llm(request, prompt_input, spinner, idata)
            else:
                await handle_enter_vector_search(request, prompt_input, spinner, idata)

        @ui.page(path)
        async def index(request: Request) -> None:
            logstuff.update_from_request(request)
            log.info(f'route triggered')

            idata = InstanceData(self.llm_config, self.env_values)

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
