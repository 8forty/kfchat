from nicegui import ui

import config


def menu() -> None:
    with ui.button(icon='menu').classes('h-12'):
        with ui.menu() as uimenu:
            ui.menu_item('Chat', lambda: ui.navigate.to('/'))
            # ui.menu_item('test api_router', lambda: ui.navigate.to('/c'))
            ui.separator()
            version_item = ui.menu_item(f'[version: {config.version}]').classes('text-xs italic')
    version_item.disable()
