import logging
import os
import sys
import tempfile
import traceback
from typing import AnyStr

from nicegui import events, ui, run
from nicegui.elements.dialog import Dialog

import logstuff
import vectorstore_chroma

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class UploadPDFDialog(Dialog):
    def __init__(self, ):
        Dialog.__init__(self)

    async def handle_upload(self, evt: events.UploadEventArguments):
        log.info(f'uploading local file {evt.name}...')
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # tmp.write(evt.content.read())
            # contents: AnyStr = await run.io_bound(evt.content.read)
            contents: AnyStr = evt.content.read()
            log.debug(f'loaded {evt.name}...')
            await run.io_bound(tmp.write, contents)
            log.debug(f'saved server file {evt.name}...')
            tmp_name = tmp.name

        try:
            log.debug(f'ingesting server file {evt.name}...')
            await run.io_bound(vectorstore_chroma.chroma.ingest_pdf, tmp_name, evt.name)
        except (Exception,) as e:
            errmsg = f'Error ingesting {evt.name}: {e}'
            traceback.print_exc(file=sys.stdout)
            log.error(errmsg)
            ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)

        os.remove(tmp_name)
        log.info(f'ingested {evt.name} via {tmp_name}')
        self.close()

    async def do_upload_pdf(self):
        self.open()
