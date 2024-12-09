from nicegui import ui

import version


def menu() -> None:
    with ui.button(icon='menu').classes('h-12'):
        with ui.menu() as uimenu:
            ui.menu_item('Chat', lambda: ui.navigate.to('/'))
            # ui.menu_item('test api_router', lambda: ui.navigate.to('/c'))
            ui.separator()
            ui.menu_item('Chromadb', lambda: ui.navigate.to('/chromadb'))
            ui.separator()
            version_item = ui.menu_item(f'[version: {version.version}]').classes('text-xs italic')
    version_item.disable()
