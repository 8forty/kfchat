import logging

from nicegui import ui

import logstuff
import version

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


def menu() -> None:
    with ui.button(icon='menu').classes('h-12'):
        with ui.menu() as uimenu:
            ui.menu_item('Chat', on_click=lambda: ui.navigate.to('/'))
            ui.separator()
            ui.menu_item('Chromadb', on_click=lambda: ui.navigate.to('/chromadb'))
            ui.separator()
            version_item = ui.menu_item(f'[version: {version.version}]').classes('text-xs italic')
            version_item.disable()
