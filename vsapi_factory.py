from vsapi import VSAPI
from vsazure import VSAzure
from vschroma import VSChroma


def create_one(api_type_name: str, parms: dict[str, str]) -> VSAPI:
    if api_type_name == 'azure':
        return VSAzure.create(api_type_name, parms)
    elif api_type_name == 'chroma':
        return VSChroma.create(api_type_name, parms)
