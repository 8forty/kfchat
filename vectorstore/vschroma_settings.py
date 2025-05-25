from basesettings import BaseSettings
from sqlitedata import FTSType
from vectorstore.vssettings import VSSettings


class VSChromaSettings(VSSettings):
    def __init__(self, init_n: int, init_fts_type: FTSType) -> None:
        super().__init__()
        self.n = init_n
        self.fts_type = init_fts_type

    @classmethod
    def from_settings(cls, rhs):
        return cls(init_n=rhs.n, init_fts_type=rhs.fts_type)

    def numbers_oneline_logging_str(self) -> str:
        return f'n:{self.n}'

    def texts_oneline_logging_str(self) -> str:
        return f''

    def specs(self) -> list[BaseSettings.SettingsSpec]:
        return [
            BaseSettings.SettingsSpec(label='n', options=[i for i in range(1, 10)], value=self.n, tooltip='number of results per query'),
            BaseSettings.SettingsSpec(label='FTS', options=[FTSType.names()], value=self.fts_type, tooltip='full-text search type'),
        ]

    async def change(self, label: str, value: any) -> None:
        if label == 'n':
            self.n = value
        elif label == 'FTS':
            self.fts_type = value
        else:
            raise ValueError(f'bad label! {label}')

