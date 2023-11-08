import dataclasses
from typing import Literal


@dataclasses.dataclass
class MLPDecoder:
    name: Literal["MLPDecoder"]
