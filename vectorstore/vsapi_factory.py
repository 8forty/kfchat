from vectorstore.vsapi import VSAPI
from vectorstore.vsazure import VSAzure
from vectorstore.vschroma import VSChroma


def create_one(vs_type_name: str, parms: dict[str, str]) -> VSAPI:
    if vs_type_name == 'azure':
        return VSAzure.create(vs_type_name, parms)
    elif vs_type_name == 'chroma':
        return VSChroma.create(vs_type_name, parms)
