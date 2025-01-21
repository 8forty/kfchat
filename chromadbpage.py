import logging
import os
import sys
import tempfile
import traceback
from typing import AnyStr

from chromadb.types import Collection
from fastapi import Request
from nicegui import ui, run, events
from nicegui.elements.dialog import Dialog

import config
import frame
import logstuff
import rbui
from vschroma import VSChroma

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class UploadFileDialog(Dialog):
    upload_id_class = 'chroma-upload'

    def __init__(self):
        Dialog.__init__(self)
        self.chunker_type: str | None = None

    async def handle_upload(self, ulargs: events.UploadEventArguments, vectorstore: VSChroma):
        local_file_name = ulargs.name
        log.info(f'uploading local file {local_file_name}...')
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            contents: AnyStr = await run.io_bound(ulargs.content.read)
            log.debug(f'loaded {local_file_name}...')
            await run.io_bound(tmpfile.write, contents)
            log.debug(f'saved file {local_file_name} to server file {tmpfile.name}...')

        collection: Collection | None = None
        try:
            log.debug(f'chunking ({self.chunker_type}) server file {tmpfile.name}...')
            # todo: configure splitter and parms
            collection = await run.io_bound(vectorstore.ingest_pdf_text_splitter, tmpfile.name, local_file_name, 1000, 200)

            if collection is None:
                errmsg = f'ingest failed for {local_file_name}'
                log.warning(errmsg)
                ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)
        except (Exception,) as e:
            errmsg = f'Error ingesting {local_file_name}: {e}'
            traceback.print_exc(file=sys.stdout)
            log.error(errmsg)
            ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)

        os.remove(tmpfile.name)
        if collection is not None:
            log.info(f'ingested {local_file_name} via {tmpfile.name}')
        self.close()

    async def do_upload_file(self, doc_type: str, chunker_type: str):
        for d in self.descendants(include_self=False):
            if self.upload_id_class in d.classes:
                if doc_type in ['pypdf', 'pymupdf']:
                    d.props(add='accept=".pdf"')
                else:
                    d.props(add='accept=".doc,.docx,.txt"')
        self.chunker_type = chunker_type  # communicates to other functions in this instance
        self.open()


def setup(path: str, pagename: str, vectorstore: VSChroma, env_values: dict[str, str]):
    @ui.refreshable
    async def chroma_ui() -> None:
        with rbui.table():
            for collection_name in await run.io_bound(vectorstore.list_index_names):
                collection = vectorstore.get_collection(collection_name)
                with rbui.tr():
                    with rbui.td(label=f'collection [{collection_name}]', td_style='width: 300px'):
                        ui.button(text='delete', on_click=lambda c=collection_name: delete_coll(c))
                        ui.separator()
                        ui.button(text='peek', on_click=lambda c=collection_name: peek(c))

                    # details table
                    with rbui.table():
                        with rbui.tr():
                            rbui.td('id')
                            rbui.td(f'{collection.id}')
                        with rbui.tr():
                            rbui.td('doc/chunk count')
                            rbui.td(f'{collection.count()}')
                        with rbui.tr():
                            rbui.td('metadata')
                            metadata_string: str = ''
                            for key in sorted(collection.metadata.keys()):
                                metadata_string += f'{key}: {collection.metadata[key]}\n'
                            rbui.td(f'{metadata_string}')
                        with rbui.tr():
                            rbui.td('configuration')
                            config_string: str = '[NOTE: these hnsw values overridden by metadata v.6+]\n'
                            for key in sorted(collection.configuration_json.keys()):
                                config_string += f'{key}: {collection.configuration_json[key]}\n'
                            rbui.td(f'{config_string}')
                        with rbui.tr():
                            rbui.td('model dimensions')
                            rbui.td(f'{collection._model.dimension}')
                        with rbui.tr():
                            rbui.td('embedding func')
                            rbui.td(f'{collection._embedding_function.__class__.__name__}\n{collection._embedding_function.__dict__}')
                        with rbui.tr():
                            rbui.td('tenant')
                            rbui.td(f'{collection.tenant}')
                        with rbui.tr():
                            rbui.td('database')
                            rbui.td(f'{collection.database}')

                        # _client, _model, _embedding_function, _data_loader, ?
                        # for key in collection.__dict__['_model'].__dict__.keys():
                        #     val = collection.__dict__['_model'].__dict__[key]
                        #     with rbui.tr():
                        #         rbui.td(f'_model.{key}')
                        #         rbui.td(str(val))

    async def delete_coll(coll_name: str) -> None:
        log.info(f'deleting collection {coll_name}')
        vectorstore.delete_index(coll_name)
        chroma_ui.refresh()

    async def peek(coll_name: str) -> None:
        collection = vectorstore.get_collection(coll_name)
        peek_n = 3  # todo: configure this?
        peeks = collection.peek(limit=peek_n)
        with ui.dialog() as peek_dialog, ui.card().classes('min-w-full'):
            with ui.column().classes('gap-y-0'):
                with ui.row():
                    ui.input(placeholder='id').props('outlined').props('color=primary').props('bg-color=white')
                    ui.button(text='id')
                ui.separator()
                with rbui.table():
                    for i in range(0, peek_n):
                        with rbui.tr():
                            doc = peeks['documents'][i][0:100] + '[...]'
                            doc_id = peeks['ids'][i]
                            rbui.td(label=f'{doc_id}')
                            rbui.td(label=f'{doc}')

        await peek_dialog
        peek_dialog.close()

    async def do_create_dialog():
        def do_create(collection_name: str, embedding_type: str):
            if len(collection_name.strip()) < 3 or len(collection_name.strip()) > 63 or len(collection_name.strip()) != len(collection_name):
                ui.notify(message='Invalid collection name', position='top', type='negative', close_button='Dismiss')
            vectorstore.create_collection(collection_name, embedding_type)
            create_dialog.submit(collection_name)

        with ui.dialog() as create_dialog, ui.card():
            ui.label('Create A Collection')
            cinput = ui.input(label='Collection Name',
                              placeholder='3-63 chars, no spaces, unders or hyphens',
                              validation={'Too short!': lambda value: len(value) >= 3,
                                          'Too long!': lambda value: len(value) <= 63,
                                          'No Spaces!': lambda value: str(value).find(' ') == -1},
                              ).classes('flex-grow').props('outlined').props('color=primary').props('bg-color=white')
            ui.button('Create: ST/all-MiniLM-L6-v2 Embedding', on_click=lambda: do_create(cinput.value, 'ST/all-MiniLM-L6-v2')).props('no-caps')
            ui.button('Create: ST/all-mpnet-base-v2 Embedding...', on_click=lambda: do_create(cinput.value, 'ST/all-mpnet-base-v2')).props('no-caps')
            ui.button('Create: OpenAI/text-embedding-3-large Embedding...', on_click=lambda: do_create(cinput.value, 'OpenAI/text-embedding-3-large')).props('no-caps')

        _ = await create_dialog
        create_dialog.close()
        create_dialog.clear()
        chroma_ui.refresh()

    @ui.page(path)
    async def index(request: Request) -> None:
        logstuff.update_from_request(request)
        log.debug(f'chromadbpage route triggered')

        with UploadFileDialog() as upload_file_dialog, ui.card():
            ui.label('Upload a File')
            ui.upload(auto_upload=True, on_upload=lambda e: upload_file_dialog.handle_upload(e, vectorstore)).classes(add=upload_file_dialog.upload_id_class)

        with frame.frame(f'{config.name} {pagename}', 'bg-white'):
            with ui.column().classes('w-full flex-grow'):  # .classes('w-full max-w-2xl mx-auto items-stretch'):
                with ui.row().classes('w-full border-solid border border-black items-center'):
                    with ui.column().classes('gap-y-2'):
                        ui.button('Refresh', on_click=lambda: chroma_ui.refresh()).props('no-caps')
                        ui.button('Create...', on_click=lambda: do_create_dialog()).props('no-caps')
                    with ui.column().classes('gap-y-2'):
                        ui.button('Add: RCTS Chunker...', on_click=lambda: upload_file_dialog.do_upload_file('pypdf', 'text')).props('no-caps')
                        ui.button('Add: Semantic Chunker...', on_click=lambda: upload_file_dialog.do_upload_file('pypdf', 'semantic')).props('no-caps')
                        # ui.button('Create: ST/all-MiniLM-L6-v2 Embedding...', on_click=lambda: upload_file_dialog.do_upload_file('pypdf', 'text')).props('no-caps')
                        # ui.button('Create: ST/all-mpnet-base-v2 Embedding...', on_click=lambda: upload_file_dialog.do_upload_file('pypdf', 'text')).props('no-caps')
                        # ui.button('Create: ST/OpenAI Embedding...', on_click=lambda: upload_file_dialog.do_upload_file('pypdf', 'semantic')).props('no-caps')
                    # with ui.column().classes('gap-y-2'):
                    #     ui.button('Upload File + Text Chunker...', on_click=lambda: upload_file_dialog.do_upload_file('pydoc', 'text')).props('no-caps')
                    #     ui.button('Upload File + Semantic Chunker...', on_click=lambda: upload_file_dialog.do_upload_file('pydoc', 'semantic')).props('no-caps')
                await chroma_ui()
