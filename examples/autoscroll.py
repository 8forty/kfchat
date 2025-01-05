import builtins

from nicegui import ui

numbers = list(range(0, 10))


def setup(path: str):
    @ui.refreshable
    async def show_numbers(scroller_class: str) -> None:
        for i in numbers:
            ui.label(f'scrolling:{i} ')

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
            with ui.card().classes('w-32 h-48'):
                sclass = 'kfname-scroll-area-scroller1'
                with ui.scroll_area().classes(add=sclass):
                    await show_numbers(sclass)


if __name__ in {'__main__', '__mp_main__'}:
    setup('/')
    ui.run(host='localhost', port=8000, show=False, reload=False)
