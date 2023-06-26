from bopt.data.superbizarre_prediction import preprocess_superbizarre_prediction_dataset, \
    preprocess_superbizarre_prediction_gold_dataset
from bopt.data.morpheme_prediction import preprocess_morpheme_prediction_dataset, \
    preprocess_morpheme_prediction_gold_dataset
from bopt.data.weibo_prediction import preprocess_weibo_prediction_dataset

preprocessors = {
    "morpheme_prediction": preprocess_morpheme_prediction_dataset,
    "morpheme_prediction_gold": preprocess_morpheme_prediction_gold_dataset,
    "superbizarre_prediction": preprocess_superbizarre_prediction_dataset,
    "superbizarre_prediction_gold": preprocess_superbizarre_prediction_gold_dataset,
    "weibo_prediction": preprocess_weibo_prediction_dataset,
}