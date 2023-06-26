from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class TokenizationSetup:
    args :Optional[Any] = None
    vocab :Optional[Any] = None
    tokenizer :Optional[Any] = None
    specials :Optional[Any] = None