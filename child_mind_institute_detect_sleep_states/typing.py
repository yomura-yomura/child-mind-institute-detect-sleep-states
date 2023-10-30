from typing import TypeAlias
import pathlib


FilePath: TypeAlias = pathlib.Path[str] | str
