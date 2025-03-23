from vectorstore.vsapi import VSAPI
from vectorstore.vschroma import VSChroma
from vectorstore.vssettings import VSSettings


def create_one(vs_type_name: str, vssettings: VSSettings, parms: dict[str, str]) -> VSAPI:
    if vs_type_name == 'chroma':
        return VSChroma.create(vs_type_name, vssettings, parms)
    # elif vs_type_name == 'azure':
    #     return VSAzure.create(vs_type_name, vssettings, parms)
