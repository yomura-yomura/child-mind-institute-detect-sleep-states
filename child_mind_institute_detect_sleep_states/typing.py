import pathlib
from typing import TypeAlias

FilePath: TypeAlias = pathlib.Path[str] | str
