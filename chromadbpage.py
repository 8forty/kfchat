import logging

import chromadb
from fastapi import Request
from nicegui import ui

import config
import logstuff
import rbui

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(config.logging_level)


def setup(path: str):
    @ui.page(path)
    async def index(request: Request) -> None:
        logstuff.update_from_request(request)
        log.debug(f'chromadbpage route triggered')
        ui.label('chromadb page')

        # with ui.column().classes('gap-y-0 w-full mx-10'):
        #     with rbui.table():
        #         with rbui.tr():  # table header row
        #             rbui.th('Your Question')
        #             rbui.th('General Info')
        #             rbui.th('Q1')
        #             rbui.th('Q2')
        #             rbui.th('Q3')
        #
        #         chat_response: ChatResponse
        #         for chat_response in chat_exchange.responses:
        #             with rbui.tr():  # table header row
        #                 rbui.td(chat_response.question)
        #                 rbui.td(chat_response.gi.gis[0].content if chat_response.gi is not None and len(chat_response.gi.gis) > 0 else ' ')
        #                 for i in range(0, 3):
        #                     if len(chat_response.qi.qis) > i:
        #                         rbui.td(chat_response.qi.qis[i].answer, tt_text=chat_response.qi.qis[i].question)
        #                     else:
        #                         rbui.td(' ')

        client = await chromadb.AsyncHttpClient(host='localhost', port=8888)
        with rbui.table():
            for coll in await client.list_collections():
                with rbui.tr():
                    rbui.td(f'collection [{coll.name}]:{coll.metadata}')
                    results: dict = await coll.query(
                        query_texts=["This is a query document about hawaii"],  # Chroma will embed this for you
                        n_results=2  # how many results to return
                    )
                    with rbui.table():
                        for key in results.keys():
                            val = results[key]
                            if key == 'documents':  # these are likely too big, just show the lengths
                                docval: str = 'lengths: ['
                                for doclist in val:
                                    docval += '[' + ','.join([str(len(doc)) for doc in doclist]) + ']\n'
                                docval = docval[0:-1] + ']'
                                val = docval
                            with rbui.tr():
                                rbui.td(key)
                                rbui.td(str(val))
