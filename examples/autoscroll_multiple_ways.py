import builtins

from nicegui import ui
from nicegui.elements.scroll_area import ScrollArea

numbers = list(range(0, 10))


def setup_manual(path: str):
    def show_numbers() -> None:
        for i in numbers:
            ui.label(f'mscrolling:{i} ')

    def add(scroller: ScrollArea) -> None:
        numbers.append(len(numbers))
        scroller.clear()
        with scroller:
            show_numbers()
        scroller.scroll_to(percent=1e6)

    @ui.page(path)
    def index():
        ui.button(text='add', on_click=lambda: add(scroller))

        with ui.row():
            with ui.card().classes('w-40 h-48'):
                with ui.scroll_area().classes() as scroller:
                    show_numbers()
                    scroller.scroll_to(percent=1e6)


def setup_refreshable_async(path: str):
    @ui.refreshable
    async def show_numbers(scroller_class: str) -> None:
        for i in numbers:
            ui.label(f'rascrolling:{i} ')

        # make sure client is connected before doing javascript
        try:
            await ui.context.client.connected(timeout=5.0)
            await ui.run_javascript(code=f"const sdiv = document.getElementsByClassName('{scroller_class}')[0].firstElementChild;"
                                         f"sdiv.scrollTop = sdiv.scrollHeight;")
            # scroller.scroll_to(percent=1.0)  # this is unpredictable, usually scrolls to penultimate item
        except builtins.TimeoutError:
            pass

    def add():
        numbers.append(len(numbers))
        show_numbers.refresh()

    @ui.page(path)
    async def index():
        ui.button(text='add', on_click=lambda: add())

        with ui.row():
            with ui.card().classes('w-40 h-48'):
                sclass = 'kfname-scroll-area-scroller1'
                with ui.scroll_area().classes(add=sclass):
                    await show_numbers(sclass)


def setup_refreshable_async2(path: str):
    @ui.refreshable
    async def show_numbers(scroller: ScrollArea) -> None:
        for i in numbers:
            ui.label(f'ra2scrolling:{i} ')

        # make sure client is connected before doing javascript
        try:
            await ui.context.client.connected(timeout=5.0)
            scroller.scroll_to(percent=1e6)
        except builtins.TimeoutError:
            pass

    def add():
        numbers.append(len(numbers))
        show_numbers.refresh()

    @ui.page(path)
    async def index():
        ui.button(text='add', on_click=lambda: add())

        with ui.row():
            with ui.card().classes('w-40 h-48'):
                with ui.scroll_area().classes() as scroller:
                    await show_numbers(scroller)


def setup_refreshable_sync(path: str):
    @ui.refreshable
    def show_numbers(scroller_class: str) -> None:
        for i in numbers:
            ui.label(f'rsscrolling:{i} ')

        # make sure client is connected before doing javascript
        try:
            # ui.context.client.connected(timeout=5.0)
            ui.run_javascript(code=f"const sdiv = document.getElementsByClassName('{scroller_class}')[0].firstElementChild;"
                                   f"sdiv.scrollTop = sdiv.scrollHeight;")
            # scroller.scroll_to(percent=1.0)  # this is unpredictable, usually scrolls to penultimate item
        except builtins.TimeoutError:
            pass

    def add():
        numbers.append(len(numbers))
        show_numbers.refresh()

    @ui.page(path)
    def index():
        ui.button(text='add', on_click=lambda: add())

        with ui.row():
            with ui.card().classes('w-40 h-48'):
                sclass = 'kfname-scroll-area-scroller1'
                with ui.scroll_area().classes(add=sclass):
                    show_numbers(sclass)


if __name__ in {'__main__', '__mp_main__'}:
    # setup_refreshable_sync('/')
    # setup_refreshable_async('/')
    # setup_refreshable_async2('/')
    setup_manual('/')
    ui.run(host='localhost', port=8000, show=False, reload=False)
