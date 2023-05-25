from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class ClassificationTrainingSetup:
    args :Optional[Any] = None
    train_dataloader :Optional[Any] = None
    train_monitor_dataloader :Optional[Any] = None
    dev_dataloader :Optional[Any] = None
    test_dataloader :Optional[Any] = None
    train_tokenization_memoizer :Optional[Any] = None
    dev_tokenization_memoizer :Optional[Any] = None
    test_tokenization_memoizer :Optional[Any] = None
    train_label_memoizer :Optional[Any] = None
    dev_label_memoizer :Optional[Any] = None
    test_label_memoizer :Optional[Any] = None
    classifier :Optional[Any] = None
    optimizer :Optional[Any] = None
    scheduler :Optional[Any] = None
    specials :Optional[Any] = None
    annealing_scheduler :Optional[Any] = None


@dataclass
class TrainingState:
    step : int
    epoch : int
    elapsed: float