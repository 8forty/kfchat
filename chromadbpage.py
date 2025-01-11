import logging

from fastapi import Request
from nicegui import ui, run

import config
import frame
import logstuff
import rbui
from vschroma import VSChroma

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)


def setup(path: str, pagename: str, vectorstore: VSChroma, env_values: dict[str, str]):
    @ui.refreshable
    async def chroma_ui() -> None:
        with rbui.table():
            for collection_name in await run.io_bound(vectorstore.list_index_names):
                with rbui.tr():
                    with rbui.td(label=f'collection [{collection_name}]', td_style='width: 300px'):
                        ui.button(text='delete', on_click=lambda c=collection_name: delete_coll(c))

                    # details table
                    with rbui.table():
                        with rbui.tr():
                            rbui.td('document/chunk count')
                            rbui.td(f'{vectorstore.count()}')
                        with rbui.tr():
                            peek_n = 3
                            rbui.td(f'peek.documents({peek_n})')
                            peek_docs = [d['documents'][0:100] + '[...]' for d in vectorstore.peek(limit=peek_n)]
                            docs = '\n-----[doc]-----\n'.join(peek_docs)  # 'ids', 'embeddings', 'metadatas', 'documents', 'data', 'uris', 'included'
                            rbui.td(f'{docs}')

                        # _client, _model, _embedding_function, _data_loader, ?
                        for key in vectorstore.collection_dict()['_model'].__dict__.keys():
                            val = vectorstore.collection_dict()['_model'].__dict__[key]
                            with rbui.tr():
                                rbui.td(f'_model.{key}')
                                rbui.td(str(val))

    async def delete_coll(coll_name: str) -> None:
        log.info(f'deleting collection {coll_name}')
        vectorstore.delete_index(coll_name)
        chroma_ui.refresh()

    @ui.page(path)
    async def index(request: Request) -> None:
        logstuff.update_from_request(request)
        log.debug(f'chromadbpage route triggered')

        with frame.frame(f'{config.name} {pagename}', 'bg-white'):
            with ui.column().classes('w-full flex-grow'):  # .classes('w-full max-w-2xl mx-auto items-stretch'):
                ui.button('refresh', on_click=lambda: chroma_ui.refresh())
                await chroma_ui()

        #
        # sample query
        #
        # sample_query = "This is a query document about hawaii"
        # sample_query_results_wanted = 2
        # results: dict = await coll.query(
        #     query_texts=[sample_query],  # Chroma will embed this for you
        #     n_results=sample_query_results_wanted
        # )
        #
        # with rbui.tr():
        #     rbui.th(label=f'Query: {sample_query} [n={sample_query_results_wanted}]', th_props='colspan="2"')
        #
        # for key in results.keys():
        #     val = results[key]
        #     if key == 'documents':  # these are likely too big, just show the lengths
        #         docval: str = 'lengths: ['
        #         for doclist in val:
        #             docval += '[' + ','.join([str(len(doc)) for doc in doclist]) + ']\n'
        #         docval = docval[0:-1] + ']'
        #         val = docval
        #     with rbui.tr():
        #         rbui.td(key)
        #         rbui.td(str(val))
