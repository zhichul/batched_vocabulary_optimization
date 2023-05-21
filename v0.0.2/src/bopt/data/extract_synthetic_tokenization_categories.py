import csv
import sys

from tqdm import tqdm

from bopt.tokenization.utils import tokens_to_tokenization, display

reader = csv.DictReader(sys.stdin, fieldnames=["id", "label", "text", "features", "segmentation"])
sp = ["[SP1]", "[SP2]", "[SP3]"]
for i, row in enumerate(tqdm(reader)):
    display(row["text"], [list(map(lambda x: x[1], filter(lambda x: x[0] not in sp,zip(row["segmentation"].split("-"), ["prefix", "stem", "suffix"]))))], [1.0]) # should not contain any specials after filtering
