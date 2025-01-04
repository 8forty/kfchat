from nicegui import ui
from nicegui.elements.scroll_area import ScrollArea

numbers = list(range(0, 10))


def setup(path: str):
    @ui.refreshable
    def show_numbers() -> None:
        for i in numbers:
            ui.label(f'scrolling:{i} ')

    def add(scroller: ScrollArea):
        numbers.append(len(numbers))
        show_numbers.refresh()
        scroller.scroll_to(percent=100.0)  # this is unpredictable with 1.0, seems better with 100.0?

    @ui.page(path)
    def index():
        ui.button(text='add', on_click=lambda: add(scroller))

        with ui.row():
            with ui.card().classes('w-32 h-48'):
                with ui.scroll_area() as scroller:
                    show_numbers()
                    scroller.scroll_to(percent=100.0)  # this works on initial display with 1.0 or 100.0?


if __name__ in {'__main__', '__mp_main__'}:
    setup('/')
    ui.run(host='localhost', port=8000, show=False, reload=False)
