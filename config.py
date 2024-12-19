import datetime
import logging
import random
import sys
import time
import timeit
import traceback

import chromadb

import logstuff

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

random.seed(27)

name = 'kfchat'

chat_exchanges_circular_list_count = 10


def now_datetime() -> str:
    return datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d-%H:%M:%S')


def now_time() -> str:
    return datetime.datetime.now(datetime.UTC).strftime('%H:%M:%S')


def secs_string(start: float, end: float = None) -> str:
    if end is None:
        end = timeit.default_timer()
    return time.strftime('%H:%M:%S', time.gmtime(end - start))


chromadb_client = None


async def make_clients_once():
    global chromadb_client
    try:
        if chromadb_client is None:
            while True:
                try:
                    # todo: configure this
                    chromadb_client = await chromadb.AsyncHttpClient(host='localhost', port=8888)
                    break
                except (Exception,) as e:
                    print(f'!!! Chroma client error, will retry in {15} secs: {e}')
                time.sleep(15)  # todo: configure this

    except (Exception,) as e:
        e = f'{sys.exc_info()[0].__name__}: {sys.exc_info()[1]}'
        log.warning(f'ERROR making client objects: {e}')
        exc = traceback.format_exc()  # sys.exc_info())
        log.warning(f'{exc}')
        raise
