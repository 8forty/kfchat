from nicegui import ui

menus = {
    'option 1': [],
    'option 2': [],
    'option 3': ['o3-so1', 'o3-so2', 'o3-so3'],
}


def mclick(text: str):
    ui.notify(f'so: {text}')


with ui.button(text='Source').props('no-caps'):
    with ui.menu():
        for k, v in menus.items():
            if len(v) == 0:
                ui.menu_item(k, on_click=lambda ceargs: mclick(ceargs.sender.default_slot.children[0].text))
            else:
                with ui.menu_item(k, auto_close=False):
                    with ui.item_section().props('side'):
                        ui.icon('keyboard_arrow_right')
                    with ui.menu().props('anchor="top end" self="top start" auto-close'):
                        for sub_k in v:
                            ui.menu_item(sub_k, on_click=lambda ceargs: mclick(ceargs.sender.default_slot.children[0].text))


        # ui.menu_item('Option 1')
        # ui.menu_item('Option 2')
        # with ui.menu_item('Option 3', auto_close=False).on('mouseenter', lambda: sub3.open()).on('mouseleave', lambda: sub3.close()):
        #     with ui.item_section().props('side'):
        #         ui.icon('keyboard_arrow_right')
        #     with ui.menu().props('anchor="top end" self="top start" auto-close') as sub3:
        #         ui.menu_item(text='Sub-option 1', on_click=lambda ceargs: so(ceargs))
        #         ui.menu_item('Sub-option 2', on_click=lambda ceargs: so(ceargs))
        #         ui.menu_item('Sub-option 3', on_click=lambda ceargs: so(ceargs))

if __name__ in {'__main__', '__mp_main__'}:
    ui.run(host='localhost', port=8000, show=False, reload=False)
