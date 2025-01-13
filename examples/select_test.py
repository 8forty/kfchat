from nicegui import ui


def setup(path: str):
    @ui.page(path)
    async def index():
        with ui.row().classes('w-full'):
            ui.select([
                'Very long option that should wrap',
                'foo',
                'bar',
                'baz',
            ]).props('popup-content-class="max-w-[100px]"')


if __name__ in {'__main__', '__mp_main__'}:
    setup('/')
    ui.run(host='localhost', port=8000, show=False, reload=False)
