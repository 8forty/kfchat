import logging
import os
import tempfile

from nicegui import events
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
        config.chat_pdf.ingest(tmp_name, evt.name)
        os.remove(tmp_name)
        log.info(f'ingested {evt.name} via {tmp_name}')

    async def do_upload_pdf(self):
        self.open()
