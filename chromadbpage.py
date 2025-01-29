import json
import logging
import os
import sys
import tempfile
import timeit
import traceback
from typing import AnyStr

from chromadb.types import Collection
from fastapi import Request
from nicegui import ui, run, events, Client
from nicegui.elements.button import Button
from nicegui.elements.dialog import Dialog
from nicegui.elements.select import Select
from nicegui.elements.spinner import Spinner

import config
import frame
import logstuff
import rbui
from lc_docloaders import docloaders
from vschroma import VSChroma

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class UploadFileDialog(Dialog):
    upload_id_class = 'chroma-upload'  # used to find the input in descendants

    def __init__(self, collection: Collection, doc_type: str, chunker_type: str):
        Dialog.__init__(self)
        self.collection = collection
        self.doc_type = doc_type
        self.chunker_type = chunker_type

    async def handle_upload(self, ulargs: events.UploadEventArguments, vectorstore: VSChroma, done_button: Button, upld_spinner: Spinner):
        done_button.set_visibility(False)
        done_button.disable()
        upld_spinner.set_visibility(True)
        try:
            local_file_name = ulargs.name
            log.info(f'uploading local file {local_file_name} for {self.collection.name}...')
            with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
                contents: AnyStr = await run.io_bound(ulargs.content.read)
                log.debug(f'loaded {local_file_name}...')
                await run.io_bound(lambda: tmpfile.write(contents))
                log.debug(f'saved file {local_file_name} to server file {tmpfile.name}...')

            try:
                log.debug(f'chunking ({self.chunker_type}) server file {tmpfile.name}...')
                await run.io_bound(lambda: vectorstore.ingest(self.collection, tmpfile.name, local_file_name, self.doc_type, self.chunker_type))
            except (Exception,) as e:
                errmsg = f'Error ingesting {local_file_name}: {e}'
                traceback.print_exc(file=sys.stdout)
                log.error(errmsg)
                ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)

            os.remove(tmpfile.name)
            log.info(f'ingested {local_file_name} via {tmpfile.name} to collection {self.collection.name}')
        finally:
            upld_spinner.set_visibility(False)
            done_button.set_visibility(True)
            done_button.enable()

    async def set_filetype_props(self):
        for d in self.descendants(include_self=False):
            if self.upload_id_class in d.classes:
                if docloaders[self.doc_type]['filetype'] == 'pdf':
                    d.props(add='accept=".pdf"')
                else:
                    d.props(add='accept=".doc,.docx,.txt"')


class CreateDialog(Dialog):
    def __init__(self):
        Dialog.__init__(self)

        with self, ui.card():
            with ui.column().classes('w-full'):
                ui.label('Create A Collection').classes('text-xl self-center')
                self.cinput = ui.input(label='Collection Name',
                                       placeholder='3-63 chars, no spaces, unders or hyphens',
                                       validation={'Too short!': lambda value: len(value) >= 3,
                                                   'Too long!': lambda value: len(value) <= 63,
                                                   'No Spaces!': lambda value: str(value).find(' ') == -1},
                                       ).classes('flex-grow min-w-80').props('outlined').props('color=primary').props('bg-color=white')
                embedding_types = VSChroma.embedding_types_list()
                et_start_value = embedding_types[0]
                etype = ui.select(label='Embedding Type:', options=embedding_types, value=et_start_value).props('square outlined label-color=green').classes('w-full')
                etype.on_value_change(lambda vc: self.select_embedding_subtype(vc.value, subtype_select))
                subtype_select = ui.select(label='Subtype:', options=[]).props('square outlined label-color=green').classes('w-full')
                self.select_embedding_subtype(et_start_value, subtype_select)
                ui.separator()
                ui.button('Create', on_click=lambda: self.create_collection_submit(self.cinput.value, etype.value, subtype_select.value)).props('no-caps').classes('self-center')

    async def create_collection_submit(self, collection_name: str, embedding_type: str, subtype: str):
        if len(collection_name.strip()) < 3 or len(collection_name.strip()) > 63 or len(collection_name.strip()) != len(collection_name):
            ui.notify(message='Invalid collection name', position='top', type='negative', close_button='Dismiss')
        else:
            log.info(f'creating collection {collection_name}: {embedding_type=} {subtype=}')
            self.submit((collection_name, embedding_type, subtype))

    @staticmethod
    def select_embedding_subtype(embedding_type: str, subtype_select: Select):
        opts = list(VSChroma.embedding_types[embedding_type].keys())
        subtype_select.set_options(opts)
        subtype_select.set_value(opts[0])

    async def reset_inputs(self):
        # todo: setting to None is an error and '' is immediate validation error
        self.cinput.set_value('')


#  upload(collection_name, doc_type, chunker_type, chunker_args, page_spinner)
class AddDialog(Dialog):
    def __init__(self):
        Dialog.__init__(self)

        with self, ui.card():
            with ui.column().classes('w-full'):
                ui.label('Add File to Collection').classes('text-xl self-center')
                doc_types = VSChroma.doc_loaders_list()
                chunker_types = VSChroma.chunkers_list()
                dtype = ui.select(label='Doc Reader:', options=doc_types, value=doc_types[0]).props('square outlined label-color=green')
                ctype = ui.select(label='Chunker:', options=chunker_types, value=chunker_types[0]).props('square outlined label-color=green')
                ui.button('Add', on_click=lambda: self.submit((dtype.value, ctype.value))).props('no-caps').classes('self-center')


# noinspection PyUnusedLocal
def setup(path: str, pagename: str, vectorstore: VSChroma, parms: dict[str, str]):
    async def delete_coll(coll_name: str) -> None:
        log.info(f'deleting collection {coll_name}')
        await run.io_bound(lambda: vectorstore.delete_index(coll_name))
        chroma_ui.refresh()

    async def peek(coll_name: str) -> None:
        collection = vectorstore.get_collection(coll_name)
        peek_n = 3  # todo: configure this?
        peeks = collection.peek(limit=peek_n)
        with ui.dialog() as peek_dialog, ui.card().classes('min-w-full'):
            with ui.column().classes('gap-y-0'):
                with rbui.table():
                    for i in range(0, peek_n):
                        with rbui.tr():
                            if 'documents' in peeks and len(peeks['documents']) > i:
                                doc_id = peeks['ids'][i]
                                doc = peeks['documents'][i][0:100] + '[...]'
                                md = '\n'.join(f'{k}:{v}' for k, v in peeks['metadatas'][i].items())
                                rbui.td(label=f'{doc_id}')
                                rbui.td(label=f'{doc}')
                                rbui.td(label=f'{md}')

        await peek_dialog
        peek_dialog.close()
        peek_dialog.clear()

    async def upload(collection_name: str, doc_type: str, chunker_type: str, page_spinner: Spinner) -> None:
        try:
            page_spinner.set_visibility(True)
            start = timeit.default_timer()
            collection = await run.io_bound(lambda: vectorstore.get_collection(collection_name))
            log.debug(f'loaded {collection_name} in {timeit.default_timer() - start:.1f}s')
        finally:
            page_spinner.set_visibility(False)

        with (UploadFileDialog(collection, doc_type, chunker_type) as upload_file_dialog, ui.card()):
            ui.label('Upload a File')
            upld = ui.upload(auto_upload=True).classes(add=upload_file_dialog.upload_id_class)
            with ui.row().classes('w-full place-content-center'):
                done_button = ui.button(text='Done', on_click=lambda: upload_file_dialog.submit(None)).props('no-caps')
                upld_spinner = ui.spinner(size='3em', type='default')
                upld_spinner.set_visibility(False)
                upld.on_upload(lambda ulargs, db=done_button: upload_file_dialog.handle_upload(ulargs, vectorstore, done_button, upld_spinner))
        await upload_file_dialog.set_filetype_props()
        await upload_file_dialog
        upload_file_dialog.close()
        upload_file_dialog.clear()
        chroma_ui.refresh()

    # noinspection PyProtectedMember
    @ui.refreshable
    async def chroma_ui(page_spinner: Spinner) -> None:
        page_spinner.set_visibility(True)

        for collection_name in await run.io_bound(vectorstore.list_index_names):
            try:
                collection_md = await run.io_bound(lambda: vectorstore.get_collection_metadata(collection_name))
            except (Exception,) as e:
                errmsg = f'Error loading metadata for collection {collection_name}: {e} (skipping)'
                log.warning(errmsg)
                ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)
                traceback.print_exc(file=sys.stdout)
                continue

            with ui.expansion().classes('w-full border-solid border border-black') as expansion:
                with expansion.add_slot('header'):
                    with ui.column().classes('w-full gap-y-0'):
                        with ui.row().classes('w-full'):
                            ui.label(collection_name).classes('min-w-32')
                            ui.label(f'{collection_md.count()}').classes('min-w-12')
                            model_name = '[unknown]'
                            efp = json.loads(collection_md.metadata['embedding_function_parms'])
                            if 'model_name' in efp:
                                model_name = efp['model_name']
                            ui.label(f'{collection_md.metadata['embedding_function_name']}')
                            ui.label(f'{model_name}')
                        with ui.row().classes('w-full gap-x-2 pt-2'):
                            ui.button(text='delete').on('click.stop', lambda c=collection_name: delete_coll(c)).props('no-caps')
                            ui.button(text='peek').on('click.stop', lambda c=collection_name: peek(c)).props('no-caps')
                            ui.button(text='add').on('click.stop', lambda c=collection_name: do_add_dialog(c, page_spinner)).props('no-caps')
                            ui.button(text='dump').on('click.stop', lambda c=collection_name: peek(c)).props('no-caps')

                # details table (embedded)
                with rbui.table():
                    with rbui.tr():
                        rbui.td(label='id', td_style='width:150px')
                        rbui.td(f'{collection_md.id}')
                    with rbui.tr():
                        rbui.td('metadata')
                        metadata_string: str = ''
                        for key in sorted(collection_md.metadata.keys()):
                            metadata_string += f'{key}: {collection_md.metadata[key]}\n'
                        rbui.td(f'{metadata_string}')
                    with rbui.tr():
                        rbui.td('configuration')
                        config_string: str = '[NOTE: THESE HNSW VALUES OVERRIDDEN BY METADATA V.6+]\n'
                        for key in sorted(collection_md.configuration_json.keys()):
                            config_string += f'{key}: {collection_md.configuration_json[key]}\n'
                        rbui.td(f'{config_string}')
                    with rbui.tr():
                        rbui.td('model dimensions')
                        # noinspection PyProtectedMember
                        rbui.td(f'{collection_md._model.dimension}')
                    with rbui.tr():
                        rbui.td('embedding func')
                        # noinspection PyProtectedMember
                        name = ''
                        extra = ''
                        # _model.base_model comes from e.g. SentenceTransformerEmbeddingFunction
                        if '_model' in collection_md._embedding_function.__dict__:
                            name = collection_md._embedding_function.__class__.__name__
                            extra += f'base_model: {collection_md._embedding_function.__dict__['_model'].model_card_data.base_model}'
                        # _model_name comes from e.g. OpenAIEmbeddingFunction
                        if '_model_name' in collection_md._embedding_function.__dict__:
                            name = collection_md._embedding_function.__class__.__name__
                            extra += f'_model_name: {collection_md._embedding_function.__dict__['_model_name']}'
                        if extra == '':
                            extra = '[need full collection, see metadata]'
                        rbui.td(f'{name}\n{extra}')
                    with rbui.tr():
                        rbui.td('tenant')
                        rbui.td(f'{collection_md.tenant}')
                    with rbui.tr():
                        rbui.td('database')
                        rbui.td(f'{collection_md.database}')

                    # rcts_args = {'chunk_size': 1000, 'chunk_overlap': 200}  # todo configure this
                    # ui.button(text=f'add pypdf+rcts:{rcts_args['chunk_size']},{rcts_args['chunk_overlap']}',
                    #           on_click=lambda c=collection_name: upload(c, PyPDFLoader.__name__, RecursiveCharacterTextSplitter.__name__, rcts_args, page_spinner)).props('no-caps')
                    # ui.button(text=f'add pymupdf+rcts:{rcts_args['chunk_size']},{rcts_args['chunk_overlap']}',
                    #           on_click=lambda c=collection_name: upload(c, PyMuPDFLoader.__name__, RecursiveCharacterTextSplitter.__name__, rcts_args, page_spinner)).props('no-caps')
                    # ui.button(text=f'add pdfminer+rcts:{rcts_args['chunk_size']},{rcts_args['chunk_overlap']}',
                    #           on_click=lambda c=collection_name: upload(c, PDFMinerLoader.__name__, RecursiveCharacterTextSplitter.__name__, rcts_args, page_spinner)).props('no-caps')
                    #
                    # ui.separator().props(sep_props)
                    # model_name = 'text-embedding-ada-002'
                    # sem_defaults_args = {'embeddings': OpenAIEmbeddings(model=model_name, openai_api_key=openai_key), }
                    # ui.button(text='add pypdf+sem(ada002):defaults',
                    #           on_click=lambda c=collection_name: upload(c, PyPDFLoader.__name__, SemanticChunker.__name__, sem_defaults_args, page_spinner)).props('no-caps')
                    #
                    # ui.separator().props(sep_props)
                    # model_name = 'text-embedding-3-small'
                    # sem_defaults_args = {'embeddings': OpenAIEmbeddings(model=model_name, openai_api_key=openai_key), }
                    # ui.button(text='add pypdf+sem(3-small):defaults',
                    #           on_click=lambda c=collection_name: upload(c, PyPDFLoader.__name__, SemanticChunker.__name__, sem_defaults_args, page_spinner)).props('no-caps')
                    #
                    # ui.separator().props(sep_props)
                    # model_name = 'text-embedding-3-small'
                    # sem_p95_args = {'embeddings': OpenAIEmbeddings(model=model_name, openai_api_key=openai_key),
                    #                 'breakpoint_threshold_type': 'percentile',
                    #                 'breakpoint_threshold_amount': 95.0, }
                    # ui.button(text='add pypdf+sem(3-small):pct,95.0',
                    #           on_click=lambda c=collection_name: upload(c, PyPDFLoader.__name__, SemanticChunker.__name__, sem_p95_args, page_spinner)).props('no-caps')
                    #
                    # ui.separator().props(sep_props)
                    # model_name = 'text-embedding-3-small'
                    # sem_sd3_args = {'embeddings': OpenAIEmbeddings(model=model_name, openai_api_key=openai_key),
                    #                 'breakpoint_threshold_type': 'standard_deviation',
                    #                 'breakpoint_threshold_amount': 3.0, }
                    # ui.button(text='add pypdf+sem(3-small):stdev,3.0',
                    #           on_click=lambda c=collection_name: upload(c, PyPDFLoader.__name__, SemanticChunker.__name__, sem_sd3_args, page_spinner)).props('no-caps')
                    #
                    # ui.separator().props(sep_props)
                    # model_name = 'text-embedding-3-small'
                    # sem_iq15_args = {'embeddings': OpenAIEmbeddings(model=model_name, openai_api_key=openai_key),
                    #                  'breakpoint_threshold_type': 'interquartile',
                    #                  'breakpoint_threshold_amount': 1.5, }
                    # ui.button(text='add pypdf+sem(3-small):iq,1.5',
                    #           on_click=lambda c=collection_name: upload(c, PyPDFLoader.__name__, SemanticChunker.__name__, sem_iq15_args, page_spinner)).props('no-caps')
                    #
                    # ui.separator().props(sep_props)
                    # model_name = 'text-embedding-3-small'
                    # sem_grad95_args = {'embeddings': OpenAIEmbeddings(model=model_name, openai_api_key=openai_key),
                    #                    'breakpoint_threshold_type': 'gradient',
                    #                    'breakpoint_threshold_amount': 95.0, }
                    # ui.button(text='add pypdf+sem(3-small):grad,95.0',
                    #           on_click=lambda c=collection_name: upload(c, PyPDFLoader.__name__, SemanticChunker.__name__, sem_grad95_args, page_spinner)).props('no-caps')

        page_spinner.set_visibility(False)

    async def do_create_dialog(create_dialog: CreateDialog, page_spinner: Spinner):
        await create_dialog.reset_inputs()
        result = await create_dialog

        page_spinner.set_visibility(True)
        if result is not None:
            (collection_name, embedding_type, subtype) = result
            await run.io_bound(lambda: vectorstore.create_collection(collection_name, embedding_type, subtype))

        # todo: page spinner
        chroma_ui.refresh()

    async def do_add_dialog(collection_name: str, page_spinner: Spinner) -> None:
        add_dialog = AddDialog()
        doc_type, chunker_type = await add_dialog
        print(f'~~~~ {doc_type=} {chunker_type=}')
        add_dialog.close()
        add_dialog.clear()
        await upload(collection_name, doc_type, chunker_type, page_spinner)
        chroma_ui.refresh()

    @ui.page(path=path)
    async def index(request: Request, client: Client) -> None:
        logstuff.update_from_request(request)
        log.debug(f'chromadbpage route triggered')

        page_spinner = ui.spinner(size='8em', type='gears').classes('absolute-center')

        # this page takes a LONG time to load sometimes, so show the spinner and "...await connection and then do the heavy computation async"
        # (per: https://github.com/zauberzeug/nicegui/discussions/2429)
        # page_spinner.visible = False
        await client.connected(timeout=30.0)

        create_dialog = CreateDialog()
        add_dialog = AddDialog()

        with frame.frame(f'{config.name} {pagename}', 'bg-white'):
            with ui.column().classes('w-full'):  # .classes('w-full max-w-2xl mx-auto items-stretch'):
                with ui.row().classes('w-full border-solid border border-black items-center'):
                    ui.button('Refresh', on_click=lambda: chroma_ui.refresh()).props('no-caps')
                    ui.button('Create...', on_click=lambda: do_create_dialog(create_dialog, page_spinner)).props('no-caps')

            with ui.scroll_area().classes('w-full flex-grow'):
                with ui.column().classes('w-full flex-grow'):  # .classes('w-full max-w-2xl mx-auto items-stretch'):
                    await chroma_ui(page_spinner)
