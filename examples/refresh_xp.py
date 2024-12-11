import random
from nicegui import ui

numbers = []


def setup(path: str):
    @ui.refreshable
    def number_ui() -> None:
        ui.label(', '.join(str(n) for n in sorted(numbers)))

    def add_number() -> None:
        numbers.append(random.randint(0, 100))
        number_ui.refresh()

    @ui.page(path)
    def index():
        number_ui()
        ui.button('Add random number', on_click=add_number)


if __name__ in {'__main__', '__mp_main__'}:
    setup('/')
    ui.run(host='localhost', port=8000, show=False, reload=False)
