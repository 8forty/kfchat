import datetime
import logging
import random
import time
import timeit
from dataclasses import dataclass
from enum import Enum

import dotenv

import logstuff
from llmdata import LLMData

log: logging.Logger = logging.getLogger(__name__)
log.setLevel(logstuff.logging_level)

dotenv.load_dotenv(override=True)
env = dotenv.dotenv_values()
