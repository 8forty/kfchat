#
# the header-frame with menu for other pages and some nicegui setup stuff
#

from contextlib import contextmanager

from nicegui import ui

from menu import menu


@contextmanager
def frame(header_title: str, bg_color: str = None):
    """Custom page frame to share the same styling and behavior across all pages"""

    # need this to get the content section to fill the space between header and footer
    #
    # https://github.com/zauberzeug/nicegui/discussions/1314
    # I just found a solution manipulating the overall page layout. By setting the .nicegui-content element to full width
    # and the .q-page element to flex layout, the content automatically grows to fill the vertical space.
    ui.query('.nicegui-content').classes('w-full px-0 mx-0')
    ui.query('.q-page').classes('flex')

    # https://github.com/zauberzeug/nicegui/discussions/3835#discussioncomment-10860335
    # NiceGUI's connection popup sets its z-index to 1000, but Quasar's footer gets a z-index of 2000.
    # That's why the [Connection lost...] popup is behind the footer.
    # As a workaround you can add this line to each page:
    ui.add_css('#popup { z-index: 10000; }')

    #  ui.colors(primary='#6E93D6', secondary='#53B689', accent='#111B1E', positive='#53B689')
    ui.colors(quasargreen='#66B969')

    bg_color = 'bg-white' if bg_color is None else bg_color
    with ui.header().classes(f'{bg_color} py-0 px-0'):
        with ui.row().classes('w-full py-2 px-2 gap-y-0'):
            menu()
            ui.space()
            ui.label(header_title).classes('text-blue text-3xl font-bold p-0 absolute-center')
            ui.space()
            ui.image('pluto.jpg').classes('w-12 place-self-end p-0 m-0').on('dblclick', lambda: ui.navigate.to('https://en.wikipedia.org/wiki/Pluto'))
        ui.separator().classes('h-1')  # .classes('w-full h-1 text-blue')
    yield
