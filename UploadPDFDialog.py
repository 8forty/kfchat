import logging
import os
import tempfile

from nicegui import events, ui
from nicegui.elements.dialog import Dialog

import config
import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


class UploadPDFDialog(Dialog):
    def __init__(self, ):
        Dialog.__init__(self)

    def handle_upload(self, evt: events.UploadEventArguments):
        log.info(f'uploading pdf {evt.name}...')
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(evt.content.read())
            tmp_name = tmp.name
        self.close()

        try:
            config.chat_pdf.ingest(tmp_name, evt.name)
        except (Exception,) as e:
            errmsg = f'Error ingesting {evt.name}: {e}'
            log.error(errmsg)
            ui.notify(message=errmsg, position='top', type='negative', close_button='Dismiss', timeout=0)

        os.remove(tmp_name)
        log.info(f'ingested {evt.name} via {tmp_name}')

    async def do_upload_pdf(self):
        self.open()
