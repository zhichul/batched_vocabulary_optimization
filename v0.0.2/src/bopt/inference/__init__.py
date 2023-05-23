from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class ClassificationInferenceSetup:
    args :Optional[Any] = None
    dataloader :Optional[Any] = None
    classifier :Optional[Any] = None
    specials :Optional[Any] = None

    # these are just placeholders to support the Setup API expected by classfier
    # they do not get filled in inference settings
    # since we only run the dataset once during inference no point in caching
    train_tokenization_memoizer :Optional[Any] = None
    dev_tokenization_memoizer :Optional[Any] = None
    test_tokenization_memoizer :Optional[Any] = None
    train_label_memoizer :Optional[Any] = None
    dev_label_memoizer :Optional[Any] = None
    test_label_memoizer :Optional[Any] = None