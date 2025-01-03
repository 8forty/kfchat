import random
from nicegui import ui
from nicegui.elements.input import Input


class PageInstanceData:
    def __init__(self):
        self.id = random.randint(0, 999)
        self.texts: list[str] = []

    # must be defined in local scope to avoid updates to all tabs/browsers
    # https://nicegui.io/documentation/refreshable#global_scope
    @ui.refreshable
    async def refresh_instance(self) -> None:
        for text in self.texts:
            ui.label(f'{self.id}: {text}')


class TestPage1:
    def __init__(self, path: str):
        async def handle_enter(prompt_input: Input, idata: PageInstanceData) -> None:
            idata.texts.append(f'{prompt_input.value.strip()}')
            idata.refresh_instance.refresh()  # NOTE: refreshes call the refresh() function on the original refresh_instance function!!!

        @ui.page(path)
        async def index() -> None:
            idata = PageInstanceData()  # this creates a "local scope" object for data and refreshes
            prompt_input = ui.input().on('keydown.enter', lambda: handle_enter(prompt_input, idata))

            await idata.refresh_instance()  # the first call is to the refresh_instance() function itself to set it up


# class TestPage2:
#     def __init__(self, path: str):
#         @ui.refreshable
#         async def refresh_cp(idata) -> None:
#             ui.label(f'refresh: {idata.id}')
#
#         async def handle_button(idata: PageInstanceData):
#             await refresh_cp(idata)
#
#         @ui.page(path)
#         async def index() -> None:
#             idata = PageInstanceData()
#             ui.label(f'{idata.id}')
#             ui.button('refresh', on_click=lambda: handle_button(idata))
#
#             await refresh_cp(idata)


if __name__ in {'__main__', '__mp_main__'}:
    cp = TestPage1('/')
    ui.run(host='localhost', port=8000, show=False, reload=False)
