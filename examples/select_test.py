from nicegui import ui


def setup(path: str):
    @ui.page(path=path)
    async def index():
        with ui.row().classes('w-full'):
            ui.select(
                label='blahblah',
                options=
                [
                    'Very long option that should wrap',
                    'foo',
                    'bar',
                    'baz',
                ]).props('outlined').classes('w-full')
            # ]).props('input-class="w-[500px]" outlined w-full')
            # ]).props('popup-content-class="max-w-[100px]"')


if __name__ in {'__main__', '__mp_main__'}:
    setup('/')
    ui.run(host='localhost', port=8000, show=False, reload=False)
