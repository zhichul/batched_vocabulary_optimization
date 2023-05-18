from bopt.data.superbizarre_prediction import preprocess_superbizarre_prediction_dataset
from bopt.data.morpheme_prediction import preprocess_morpheme_prediction_dataset

preprocessors = {
    "morpheme_prediction": preprocess_morpheme_prediction_dataset,
    "superbizarre_prediction": preprocess_superbizarre_prediction_dataset,
}