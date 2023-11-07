from typing import Literal
import dataclasses


@dataclasses.dataclass
class MLPDecoder:
    name: Literal["MLPDecoder"]
