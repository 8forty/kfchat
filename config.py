import datetime
import logging
import random
import time
import timeit

from chatpdf import ChatPDF

random.seed(27)

name = 'kfchat'

chat_exchanges_circular_list_count = 10

# todo: this should be moved
chat_pdf = ChatPDF()


def now_datetime() -> str:
    return datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d-%H:%M:%S')


def now_time() -> str:
    return datetime.datetime.now(datetime.UTC).strftime('%H:%M:%S')


def secs_string(start: float, end: float = None) -> str:
    if end is None:
        end = timeit.default_timer()
    return time.strftime('%H:%M:%S', time.gmtime(end - start))
