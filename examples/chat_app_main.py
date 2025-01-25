#!/usr/bin/env python3
from datetime import datetime, UTC
from typing import List, Tuple
from uuid import uuid4

from fastapi import FastAPI
from nicegui import ui

messages: List[Tuple[str, str, str, str]] = []

app = FastAPI()


@ui.refreshable
def chat_messages(own_id: str) -> None:
    if messages:
        for user_id, avatar, text, stamp in messages:
            ui.chat_message(text=text, stamp=stamp, avatar=avatar, sent=own_id == user_id)
    else:
        ui.label('No messages yet').classes('mx-auto my-36')
    ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')


@ui.page(path='/')
async def setup():
    def handle_enter() -> None:
        stamp = datetime.now(UTC).strftime('%X')
        messages.append((user_id, avatar, text.value, stamp))
        text.value = ''
        chat_messages.refresh()

    user_id = str(uuid4())
    avatar = f'https://robohash.org/{user_id}?bgset=bg2'

    ui.add_css(r'a:link, a:visited {color: inherit !important; text-decoration: none; font-weight: 500}')
    with ui.footer().classes('bg-white'), ui.column().classes('w-full max-w-3xl mx-auto my-6'):
        with ui.row().classes('w-full no-wrap items-center'):
            with ui.avatar().on('click', lambda: ui.navigate.to(setup)):
                ui.image(avatar)
            text = ui.input(placeholder='message').on('keydown.enter', handle_enter)
            text.props('rounded outlined input-class=mx-3').classes('flex-grow')

    await ui.context.client.connected()  # chat_messages(...) uses run_javascript which is only possible after connecting
    with ui.column().classes('w-full max-w-2xl mx-auto items-stretch'):
        chat_messages(user_id)


def init(fastapi_app: FastAPI) -> None:
    ui.run_with(fastapi_app, storage_secret='pick your private secret here')


init(app)
