import sys
import math
from bopt.data.utils import load_weights, normalized_weights

file1 = sys.argv[1]
file2 = sys.argv[2]
n = int(sys.argv[3])

d1 = normalized_weights(load_weights(file1))
d2 = normalized_weights(load_weights(file2))



shared = set(d1.keys()) & set(d2.keys())
shared.remove("[UNK]")
shared.remove("[PAD]")
shared.remove("[SP1]")
shared.remove("[SP2]")
shared.remove("[SP3]")
shared.remove("[SP4]")
shared.remove("[SP5]")
shared.remove("[SEP]")
shared.remove("[WBD]")
shared.remove("[MASK]")
shared.remove("[CLS]")
shared.remove("[EOS]")
shared.remove("[BOS]")
shared.remove("<unk>")

up_in_1 = sorted([(piece, math.exp(d1[piece]) - math.exp(d2[piece])) for piece in shared], reverse=True, key=lambda x: x[1])
up_in_2 = sorted([(piece, math.exp(d2[piece]) - math.exp(d1[piece])) for piece in shared], reverse=True, key=lambda x: x[1])
print("m1", max([v for p,v in up_in_1]))
print("m2", max([v for p,v in up_in_2]))
print(sum(abs(math.exp(d1[k]) - math.exp(d2[k])) for k in shared))
print(max(d1.values()))
print(max(d2.values()))
print(file1)
print("\n".join([f"{p[0]} {round(p[1],4)}" for p in up_in_1[:n]]))
print(file2)
print("\n".join([f"{p[0]} {round(p[1],10)}" for p in up_in_2[:n]]))
