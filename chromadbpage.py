import json
import logging
import os
import sys
import tempfile
import timeit
import traceback

import chromadb
from chromadb.types import Collection
from fastapi import Request
from nicegui import ui, run, events, Client
from nicegui.elements.button import Button
from nicegui.elements.dialog import Dialog
from nicegui.elements.select import Select
from nicegui.elements.spinner import Spinner
from openai.types import Upload

import config
import frame
import logstuff
import rbui
from config import FTSType
from langchain.lc_docloaders import docloaders
from vectorstore.vschroma import VSChroma

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


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
                                       ).classes('flex-grow min-w-80').props('outlined')

                # todo: allow diff't selections
                fts_types_all = 'All'  # f'All ({', '.join(FTSType.names())})'
                fts_types = ui.select(label='FTS Types:', options=[fts_types_all], value=fts_types_all).props('square outlined label-color=green').classes('w-full')

                embedding_types = VSChroma.embedding_types_list()
                et_start_value = embedding_types[0]
                embedding_type = ui.select(label='Embedding Type:', options=embedding_types, value=et_start_value).props('square outlined label-color=green').classes('w-full')
                embedding_type.on_value_change(lambda vc: self.select_embedding_subtype(vc.value, subtype_select))
                subtype_select = ui.select(label='Subtype:', options=[]).props('square outlined label-color=green').classes('w-full')
                self.select_embedding_subtype(et_start_value, subtype_select)
                ui.separator()
                ui.button('Create',
                          on_click=lambda: self.create_collection_submit(self.cinput.value, FTSType.members(), embedding_type.value, subtype_select.value)).props('no-caps').classes('self-center')

    async def create_collection_submit(self, collection_name: str, fts_types: list[FTSType], embedding_type: str, subtype: str):
        if len(collection_name.strip()) < 3 or len(collection_name.strip()) > 63 or len(collection_name.strip()) != len(collection_name):
            ui.notify(message='Invalid collection name', position='top', type='negative', close_button='Dismiss')
        else:
            log.info(f'creating collection {collection_name}: {embedding_type=} {subtype=}')
            self.submit((collection_name, fts_types, embedding_type, subtype))

    @staticmethod
    def select_embedding_subtype(embedding_type: str, subtype_select: Select):
        opts = list(VSChroma.embedding_types_list(embedding_type))
        subtype_select.set_options(opts)
        subtype_select.set_value(opts[0])

    async def reset_inputs(self):
        # todo: setting to None is an error and '' is immediate validation error
        self.cinput.set_value('')


class UploadFileDialog(Dialog):

    def __init__(self, collection: Collection):
        Dialog.__init__(self)
        self.collection = collection

    async def handle_upload(self, ulargs: events.UploadEventArguments, docloader_type: str, chunker_type: str, vectorstore: VSChroma,
                            done_button: Button, upld_spinner: Spinner):
        done_button.set_visibility(False)
        done_button.disable()
        upld_spinner.set_visibility(True)
        try:
            local_file_name = ulargs.name
            log.info(f'uploading local file {local_file_name} for {self.collection.name}...')
            with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
                contents = await run.io_bound(ulargs.content.read)
                log.debug(f'loaded {local_file_name}...')
                await run.io_bound(lambda: tmpfile.write(contents))
                log.debug(f'saved file {local_file_name} to server file {tmpfile.name}...')

            errmsg = None
            try:
                log.debug(f'chunking ({chunker_type}) server file {tmpfile.name}...')
                await run.io_bound(lambda: vectorstore.ingest(self.collection, tmpfile.name, local_file_name, docloader_type, chunker_type))
            except (Exception,) as e:
                errmsg = f'Error ingesting {local_file_name}: {e.__class__.__name__}: {e}'
                if not isinstance(e, VSChroma.EmptyIngestError) and not isinstance(e, VSChroma.OllamaEmbeddingsError):
                    traceback.print_exc(file=sys.stdout)
                log.error(errmsg)
                ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)

            try:
                os.remove(tmpfile.name)
            finally:
                if errmsg is None:
                    log.info(f'ingested {local_file_name} via {tmpfile.name} to collection {self.collection.name}')
        finally:
            upld_spinner.set_visibility(False)
            done_button.set_visibility(True)
            done_button.enable()


# noinspection PyUnusedLocal
def setup(path: str, pagename: str, vectorstore: VSChroma, parms: dict[str, str]):
    async def do_create_dialog(create_dialog: CreateDialog, page_spinner: Spinner):
        try:
            await create_dialog.reset_inputs()
            result = await create_dialog

            page_spinner.set_visibility(True)
            try:
                if result is not None:
                    (collection_name, fts_types, embedding_type, subtype) = result
                    await run.io_bound(lambda: vectorstore.create_collection(collection_name, fts_types, embedding_type, subtype))
            except (Exception,) as e:
                page_spinner.set_visibility(False)
                raise e
        except (Exception,) as e:
            errmsg = f'Error creating collection: {e.__class__.__name__}: {e}'
            traceback.print_exc(file=sys.stdout)
            log.warning(errmsg)
            ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)

        # todo: page spinner
        chroma_ui.refresh()

    async def do_delete_collection(coll_name: str) -> None:
        log.info(f'deleting collection {coll_name}')
        await run.io_bound(lambda: vectorstore.delete_collection(coll_name))
        chroma_ui.refresh()

    async def do_peek_dialog(coll_name: str) -> None:
        collection = vectorstore.get_collection(coll_name)
        peek_n = 3  # todo: configure this?
        peeks = collection.peek(limit=peek_n)
        with ui.dialog() as peek_dialog, ui.card().classes('min-w-full'):
            with ui.column().classes('gap-y-0 w-full'):
                ui.label(f'Peek: {coll_name}:')
                with rbui.table():
                    with rbui.tr():
                        rbui.th('ID')
                        rbui.th('Chunk')
                        rbui.th('Metadata')
                    for i in range(0, peek_n):
                        with rbui.tr():
                            if 'documents' in peeks and len(peeks['documents']) > i:
                                doc_id = peeks['ids'][i]
                                doc = peeks['documents'][i][0:300] + '[...]'
                                md = '\n'.join(f'{k}:{v}' for k, v in peeks['metadatas'][i].items())
                                rbui.td(label=f'{doc_id}')
                                rbui.td(label=f'{doc}')
                                rbui.td(label=f'{md}')

        await peek_dialog
        peek_dialog.close()
        peek_dialog.clear()

    async def do_dump(coll_name: str) -> None:
        collection = vectorstore.get_collection(coll_name)
        offset = 0
        max_read_count = 50
        write_count = 0
        with tempfile.NamedTemporaryFile(prefix=f'{coll_name}_', suffix='.csv', mode='w+', delete=False) as tfile:
            while True:
                gr: chromadb.GetResult = collection.get(offset=offset, limit=max_read_count, include=['documents'])
                for chunk_id, doc in zip(gr["ids"], gr["documents"]):
                    # options for a dump that's essentially for debugging
                    doc = doc.encode('ascii', errors='replace').decode('ascii')
                    doc = doc.replace('"', "'")

                    tfile.write(f'\n{chunk_id}: "{doc}"\n')
                    write_count += 1
                if len(gr['ids']) < max_read_count:
                    break
                offset += len(gr['ids'])

        if write_count > 0:
            msg = f'dumped {write_count} documents from collection {coll_name} to server file {tfile.name}'
            log.info(msg)
            ui.notify(message=msg, position='top', type='info')

    async def do_add_dialog(collection_name: str, page_spinner: Spinner) -> None:

        def set_filetype_props(upload_element: Upload, doc_type: str):
            types = ','.join(f'.{ft}' for ft in docloaders[doc_type]['filetypes'])
            upload_element.props(remove='accept')
            upload_element.props(add=f'accept="{types}"')

        try:
            page_spinner.set_visibility(True)
            start = timeit.default_timer()
            collection = await run.io_bound(lambda: vectorstore.get_collection(collection_name))
            log.debug(f'loaded {collection_name} in {timeit.default_timer() - start:.1f}s')
        except (Exception,) as e:
            errmsg = f'Error loading collection: {collection_name}: {e.__class__.__name__}: {e}'
            log.warning(errmsg)
            ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)
            raise e
        finally:
            page_spinner.set_visibility(False)

        with UploadFileDialog(collection) as upload_file_dialog, ui.card():
            with ui.column().classes('w-full'):
                ui.label(f'Add File to {collection_name}').classes('text-xl self-center')
                docloaders_list = VSChroma.doc_loaders_list()
                chunker_types = VSChroma.chunkers_list()
                docreader_type = ui.select(label='Doc Reader:', options=docloaders_list,
                                           on_change=lambda vcargs: set_filetype_props(upld, vcargs.value), value=docloaders_list[0]).props('square outlined label-color=green')
                chunker_type = ui.select(label='Chunker:', options=chunker_types, value=chunker_types[0]).props('square outlined label-color=green')
                ui.label('FTS Types: all').classes('italic')
                ui.separator()
                ui.label('Choose a file:').classes('text-quasargreen')
                upld = ui.upload(auto_upload=True)
                set_filetype_props(upld, docreader_type.value)
                with ui.row().classes('w-full place-content-center'):
                    done_button = ui.button(text='Done', on_click=lambda: upload_file_dialog.submit(None)).props('no-caps')
                    upld_spinner = ui.spinner(size='3em', type='default')
                    upld_spinner.set_visibility(False)
                    # todo: push fts_types value up to config/settings
                    upld.on_upload(lambda ulargs, db=done_button: upload_file_dialog.handle_upload(ulargs,
                                                                                                   docreader_type.value, chunker_type.value,
                                                                                                   vectorstore, done_button, upld_spinner))

        await upload_file_dialog
        upload_file_dialog.close()
        upload_file_dialog.clear()
        chroma_ui.refresh()

    # noinspection PyProtectedMember
    @ui.refreshable
    async def chroma_ui(page_spinner: Spinner) -> None:
        page_spinner.set_visibility(True)

        colls_with_md = []
        for collection_name in await run.io_bound(vectorstore.list_collection_names):
            try:
                colls_with_md.append(await run.io_bound(lambda: vectorstore.get_partial_collection(collection_name)))
            except (Exception,) as e:
                errmsg = f'Error loading metadata for collection {collection_name}: {e.__class__.__name__}: {e} (skipping)'
                log.warning(errmsg)
                ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)
                traceback.print_exc(file=sys.stdout)
                continue
        ancient = config.ancient_datetime()
        colls_with_md = sorted(colls_with_md, key=lambda coll: coll.metadata['created'] if 'created' in coll.metadata else ancient, reverse=True)  # newest on top

        for collection_md in colls_with_md:
            with ui.expansion().classes('w-full border-solid border border-white') as expansion:
                with expansion.add_slot('header'):
                    with ui.column().classes('w-full gap-y-0'):
                        with ui.row().classes('w-full grid grid-cols-6 grid-rows-1'):
                            ui.label(collection_md.name).classes('text-orange-500 font-bold')
                            ui.label(f'[{collection_md.count()} chunks]')
                            ui.label(f'{collection_md.metadata['embedding_type'] if 'embedding_type' in collection_md.metadata else 'embedding-type:unknown'}')
                            efp = json.loads(collection_md.metadata['embedding_function_parms'])
                            ui.label(f'{efp['model_name'] if 'model_name' in efp else 'model:unknown'}')
                            ui.label(f'created:{collection_md.metadata['created'] if 'created' in collection_md.metadata else 'unknown'}').classes('italic')
                        with ui.row().classes('w-full gap-x-2 pt-2'):
                            ui.button(text='delete').on('click.stop', lambda c=collection_md.name: do_delete_collection(c)).props('no-caps')
                            ui.button(text='peek').on('click.stop', lambda c=collection_md.name: do_peek_dialog(c)).props('no-caps')
                            ui.button(text='add').on('click.stop', lambda c=collection_md.name: do_add_dialog(c, page_spinner)).props('no-caps')
                            ui.button(text='dump').on('click.stop', lambda c=collection_md.name: do_dump(c)).props('no-caps')

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

        page_spinner.set_visibility(False)

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

        with frame.frame(f'{config.name} {pagename}'):
            with ui.column().classes('w-full'):  # .classes('w-full max-w-2xl mx-auto items-stretch'):
                with ui.row().classes('w-full border-solid border border-white items-center'):
                    ui.button('Refresh', on_click=lambda: chroma_ui.refresh()).props('no-caps')
                    ui.button('Create...', on_click=lambda: do_create_dialog(create_dialog, page_spinner)).props('no-caps')

            with ui.scroll_area().classes('w-full flex-grow'):
                with ui.column().classes('w-full flex-grow'):  # .classes('w-full max-w-2xl mx-auto items-stretch'):
                    await chroma_ui(page_spinner)
