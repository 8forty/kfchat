import logging
import random

import dotenv

import logstuff
from llmdata import LLMData
from sqlitedata import SQLiteData

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

random.seed(27)

dotenv.load_dotenv(override=True)
env = dotenv.dotenv_values()

name = 'kfchat'

chat_exchanges_circular_list_count = 10

sql_path = 'c:/sqlite/kfchat/kfchat.sqlite3'  # slashes work ok on windows

sql_data = SQLiteData()

llm_data = LLMData()
