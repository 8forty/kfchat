from nicegui import ui
from nicegui.elements.scroll_area import ScrollArea

numbers = list(range(0, 10))


def setup(path: str):
    async def show_numbers(scroller: ScrollArea) -> None:
        for i in numbers:
            ui.label(f'scrolling:{i} ')
        scroller.scroll_to(percent=1e6)

    async def add(scroller: ScrollArea) -> None:
        numbers.append(len(numbers))
        scroller.clear()
        with scroller:
            await show_numbers(scroller)

    @ui.page(path=path)
    async def index():
        ui.button(text='add', on_click=lambda: add(scroller))

        with ui.row():
            with ui.card().classes('w-40 h-48'):
                with ui.scroll_area().classes() as scroller:
                    await show_numbers(scroller)


if __name__ in {'__main__', '__mp_main__'}:
    setup('/')
    ui.run(host='localhost', port=8000, show=False, reload=False)
