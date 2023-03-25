import code

from tqdm import tqdm

from bopt.data.language_modeling.unigram import LanguageModelingUnigramDataset
from bopt.data.morpheme_prediction.unigram import MorphemePredictionUnigramDataset

import sys

from bopt.data.sentiment_analysis.unigram import SentimentAnalysisUnigramDataset

task = sys.argv[1]
cache_root = sys.argv[2]

if task == "morpheme_prediction":
    d = MorphemePredictionUnigramDataset(cache_root)
    used_ids = set(id.item() for i in tqdm(range(len(d))) for id in d[i][0])
    print(len(used_ids))
if task == "sentiment_analysis":
    d = SentimentAnalysisUnigramDataset(cache_root)
    used_ids = set(id.item() for i in tqdm(range(len(d))) for id in d[i][0])
    print(len(used_ids))
if task == "language_modeling":
    d = LanguageModelingUnigramDataset(cache_root)
    used_ids = set(id.item() for i in tqdm(range(len(d))) for id in d[i][0])
    print(len(used_ids))


