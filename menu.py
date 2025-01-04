import logging

from nicegui import ui

import logstuff
import version
from uploadpdfdialog import UploadPDFDialog

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


def menu() -> None:
    with UploadPDFDialog() as upload_pdf_dialog, ui.card():
        ui.label('Upload a PDF')
        ui.upload(auto_upload=True, on_upload=lambda e: upload_pdf_dialog.handle_upload(e)).props('accept=".pdf"')

    with ui.button(icon='menu').classes('h-12'):
        with ui.menu() as uimenu:
            ui.menu_item('Chat', on_click=lambda: ui.navigate.to('/'))
            ui.menu_item('Upload PDF...', on_click=lambda: upload_pdf_dialog.do_upload_pdf())
            ui.separator()
            ui.menu_item('Chromadb', on_click=lambda: ui.navigate.to('/chromadb'))
            ui.separator()
            version_item = ui.menu_item(f'[version: {version.version}]').classes('text-xs italic')
            version_item.disable()
