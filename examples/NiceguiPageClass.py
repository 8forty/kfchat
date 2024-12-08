import builtins
import random
from typing import Literal

from fastapi import Request
from nicegui import ui, app


class NiceguiPageClass:
    class PageInstanceData:
        def __init__(self):
            # this key is unique to each page b/c it's initialized by the @ui.page function
            self.unique_id = random.randint(0, 999)

    def __init__(self):
        # this key is shared by all instances of NiceguiPageClass
        self.shared_key = random.randint(0, 999)

        @ui.refreshable
        async def refresh_npc(npc) -> None:
            ui.label(f'refresh: {self.shared_key} {npc.unique_id}')

        async def handle_button(npc: NiceguiPageClass.PageInstanceData):
            await refresh_npc(npc)

        @ui.page('/')
        async def index(request: Request) -> None:

            # unique instance data for each incoming top-level connection
            npc = NiceguiPageClass.PageInstanceData()

            shared_key = f'sss_{self.shared_key}'
            key = f'uuu_{npc.unique_id}'

            # check the key shared by all instances
            # NOTE: it's "old" on duplicate tabs on Chrome & Edge & ??? (all browsers maybe?), resulting in the need for separate instance data
            sharedkeytext = f'{shared_key}: '
            if shared_key not in app.storage.tab:
                app.storage.tab[shared_key] = 27  # whatever
                sharedkeytext += ' new'
            else:
                sharedkeytext += ' old'

            # check the unique page-instance key
            keytext = f'{key}: '
            if key not in app.storage.tab:
                app.storage.tab[key] = npc
                keytext += ' new'
            else:
                keytext += ' old'

            print(f'~~~~ {sharedkeytext}')
            print(f'~~~~ {keytext}')
            ui.label(sharedkeytext)
            ui.label(keytext)
            ui.button('refresh', on_click=lambda: handle_button(npc))

            await refresh_npc(npc)
