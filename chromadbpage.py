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


class UploadPDFDialog(Dialog):
    def __init__(self):
        Dialog.__init__(self)

    async def handle_upload(self, evt: events.UploadEventArguments, vectorstore: VSChroma):
        log.info(f'uploading local file {evt.name}...')
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # tmp.write(evt.content.read())
            contents: AnyStr = await run.io_bound(evt.content.read)
            # contents: AnyStr = evt.content.read()
            log.debug(f'loaded {evt.name}...')
            await run.io_bound(tmp.write, contents)
            log.debug(f'saved file {evt.name} to server file {tmp.name}...')

        collection: Collection = None
        try:
            log.debug(f'ingesting server file {tmp.name}...')
            collection = await run.io_bound(vectorstore.ingest_pdf, tmp.name, evt.name)

            if collection is None:
                errmsg = f'ingest failed for {evt.name}'
                log.warning(errmsg)
                ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)
        except (Exception,) as e:
            errmsg = f'Error ingesting {evt.name}: {e}'
            traceback.print_exc(file=sys.stdout)
            log.error(errmsg)
            ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)

        os.remove(tmp.name)
        if collection is not None:
            log.info(f'ingested {evt.name} via {tmp.name}')
        self.close()

    async def do_upload_pdf(self):
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

                    # details table
                    with rbui.table():
                        with rbui.tr():
                            rbui.td('document/chunk count')
                            rbui.td(f'{collection.count()}')
                        with rbui.tr():
                            peek_n = 3
                            rbui.td(f'peek.documents({peek_n})')
                            peek_docs = [d[0:100] + '[...]' for d in collection.peek(limit=peek_n)['documents']]
                            docs = '\n-----[doc]-----\n'.join(peek_docs)  # 'ids', 'embeddings', 'metadatas', 'documents', 'data', 'uris', 'included'
                            rbui.td(f'{docs}')

                        # _client, _model, _embedding_function, _data_loader, ?
                        for key in collection.__dict__['_model'].__dict__.keys():
                            val = collection.__dict__['_model'].__dict__[key]
                            with rbui.tr():
                                rbui.td(f'_model.{key}')
                                rbui.td(str(val))

    async def delete_coll(coll_name: str) -> None:
        log.info(f'deleting collection {coll_name}')
        vectorstore.delete_index(coll_name)
        chroma_ui.refresh()

    @ui.page(path)
    async def index(request: Request) -> None:
        logstuff.update_from_request(request)
        log.debug(f'chromadbpage route triggered')

        with UploadPDFDialog() as upload_pdf_dialog, ui.card():
            ui.label('Upload a PDF')
            ui.upload(auto_upload=True, on_upload=lambda e: upload_pdf_dialog.handle_upload(e, vectorstore)).props('accept=".pdf"')

        with frame.frame(f'{config.name} {pagename}', 'bg-white'):
            with ui.column().classes('w-full flex-grow'):  # .classes('w-full max-w-2xl mx-auto items-stretch'):
                with ui.row().classes('w-full border-solid border border-black'):
                    ui.button('Upload PDF...', on_click=lambda: upload_pdf_dialog.do_upload_pdf())
                    ui.button('refresh', on_click=lambda: chroma_ui.refresh())
                await chroma_ui()
