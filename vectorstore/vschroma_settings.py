from config import FTSType
from vectorstore.vssettings import VSSettings


class VSChromaSettings(VSSettings):
    def __init__(self, init_n: int, init_fts_type: FTSType) -> None:
        super().__init__(init_n)
        self.fts_type = init_fts_type

    def __repr__(self) -> str:
        return f'{self.__class__!s}:{self.__dict__!r}'
        # return f'{self.__class__!s}:{self.result_id=!r},{self.metrics=!r},self.content="{self.content[0:20]}{"..." if len(self.content) > 20 else ""}"'

    @classmethod
    def from_settings(cls, rhs):
        return cls(init_n=rhs.n, init_fts_type=rhs.fts_type)

    def numbers_oneline_logging_str(self) -> str:
        return f'n:{self.n}, fts_type:{self.fts_type}'
