import sys
from typing import Tuple, List

from bopt.superbizarre.derivator import Derivator
from bopt.tokenization.utils import display

if __name__ == "__main__":
    dr = Derivator()
    for line in sys.stdin:
        word = line.strip()
        derivation: Tuple[List[str], str, List[str]] = dr.derive(word, mode="morphemes")
        pieces = derivation[0] + [word[sum(len(p) for p in derivation[0]):len(word)-sum(len(s) for s in derivation[2])]]+ derivation[2]
        cats = ["prefix"] * len(derivation[0]) + ["stem"] + ["suffix"] * len(derivation[2])
        if "".join(pieces) == word:
            display(word, [cats], [1.0])