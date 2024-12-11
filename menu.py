import logging
import os
import tempfile

from nicegui import ui, events
from nicegui.elements.dialog import Dialog

import config
import version
from chatpdf import ChatPDF

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(config.logging_level)

# todo: this should be moved
chat_pdf = ChatPDF()


def handle_upload_pdf(upload_dialog: Dialog, evt: events.UploadEventArguments):
    log.info(f'uploading pdf {evt.name}...')
    tmp_name = None
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(evt.content.read())
        tmp_name = tmp.name
    upload_dialog.close()
    chat_pdf.ingest(tmp_name, evt.name)
    os.remove(tmp_name)
    log.info(f'ingested {evt.name} via {tmp_name}')


async def do_upload_pdf(upload_dialog: Dialog):
    upload_dialog.open()


def menu() -> None:
    with ui.dialog() as upload_pdf_dialog, ui.card():
        ui.label('Upload a PDF')
        ui.upload(auto_upload=True, on_upload=lambda e: handle_upload_pdf(upload_pdf_dialog, e)).props('accept=".pdf"')

    with ui.button(icon='menu').classes('h-12'):
        with ui.menu() as uimenu:
            ui.menu_item('Chat', on_click=lambda: ui.navigate.to('/'))
            ui.menu_item('Upload PDF...', on_click=lambda: do_upload_pdf(upload_pdf_dialog))
            ui.separator()
            ui.menu_item('Chromadb', on_click=lambda: ui.navigate.to('/chromadb'))
            ui.separator()
            version_item = ui.menu_item(f'[version: {version.version}]').classes('text-xs italic')
            version_item.disable()
