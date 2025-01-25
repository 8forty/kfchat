import asyncio

from nicegui import ui


def setup(path: str):
    async def do_async_stuff(selected: str):
        await asyncio.sleep(2)
        print(f'selected: {selected}')

    @ui.page(path=path)
    async def index():
        with ui.row().classes('w-full'):
            ui.select(label='LL:',
                      options=['abc', 'def', 'xyz'],
                      value='abc'
                      ).on_value_change(callback=lambda vc: do_async_stuff(vc.value))


if __name__ in {'__main__', '__mp_main__'}:
    setup('/')
    ui.run(host='localhost', port=8000, show=False, reload=False)
