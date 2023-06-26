from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class InnerLoopOutput:
    params: Optional[Any] = None
    one_step_params: Optional[Any] = None
    one_step_loss: Optional[Any] = None
    init_tokenizer_params: Optional[Any] = None
    fmodel: Optional[Any] = None
    buffers_first: Optional[Any] = None
    buffers_second: Optional[Any] = None
    batch_ids: Optional[Any] = None
    last_step: Optional[Any] = None
    last_grad: Optional[Any] = None
    zero_iterate: Optional[Any] = None
    first_iterate: Optional[Any] = None
    second_iterate: Optional[Any] = None
    third_iterate: Optional[Any] = None
    g_1: Optional[Any] = None
    g_2: Optional[Any] = None
    second_step:  Optional[Any] = None
    first_step:  Optional[Any] = None
    logline: Optional[Any] = None


@dataclass
class ClassificationBilevelTrainingSetup:
    args :Optional[Any] = None
    train_inner_dataloader :Optional[Any] = None
    train_outer_dataloader :Optional[Any] = None
    train_inner_monitor_dataloader :Optional[Any] = None
    train_outer_monitor_dataloader :Optional[Any] = None
    dev_dataloader :Optional[Any] = None
    test_dataloader :Optional[Any] = None
    train_inner_tokenization_memoizer :Optional[Any] = None
    train_outer_tokenization_memoizer :Optional[Any] = None
    dev_tokenization_memoizer :Optional[Any] = None
    test_tokenization_memoizer :Optional[Any] = None
    train_inner_label_memoizer :Optional[Any] = None
    train_outer_label_memoizer :Optional[Any] = None
    dev_label_memoizer :Optional[Any] = None
    test_label_memoizer :Optional[Any] = None
    classifier :Optional[Any] = None
    inner_optimizer :Optional[Any] = None
    outer_optimizer :Optional[Any] = None
    inner_scheduler :Optional[Any] = None
    outer_scheduler :Optional[Any] = None
    specials :Optional[Any] = None
    annealing_scheduler :Optional[Any] = None
    optimizer_builder: Optional[Any] = None


@dataclass
class TrainingState:
    step : int
    epoch : int
    elapsed: float