import logging
import sys
from dataclasses import dataclass

from fastapi import Request

import config


@dataclass
class LogPrefixData:
    version: str = config.version
    host: str | None = None
    client_ip: str | None = None

    def localhost(self) -> bool:
        return self.host is not None and (self.host.startswith('localhost') or self.host.startswith('127.0.0.1'))

    def __str__(self):
        return f'v[{self.version}] host[{self.host}] client[{self.client_ip}]'


# this let's us have a custom prefix
class LogPrefixFilter(logging.Filter):
    def filter(self, record):
        if log_prefix_data.host is None or log_prefix_data.localhost():
            record.log_prefix_data = '[...]'
        else:
            record.log_prefix_data = str(log_prefix_data)
        return True


def update_from_request(request: Request):
    headers = request.headers
    # todo: configure these
    log_prefix_data.host = headers.get('Host', 'unknown')
    log_prefix_data.client_ip = headers.get('X-Client-Ip', 'unknown')


# configures logging for everthing in the module

log_prefix_data = LogPrefixData()
log_prefix_filter = LogPrefixFilter()
logging.basicConfig(
    level=logging.WARNING,
    format=f'%(asctime)s [%(levelname)s] [%(name)s:%(funcName)s] %(log_prefix_data)s %(message)s',
    datefmt='%H:%M:%S' if log_prefix_data.localhost() else '%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
    force=True
)
# add the filter to the handler(s) to avoid KeyError on the custom format i.e. log_prefix_data
# https://stackoverflow.com/questions/17275334/what-is-a-correct-way-to-filter-different-loggers-using-python-logging
for handler in logging.root.handlers:
    handler.addFilter(log_prefix_filter)
