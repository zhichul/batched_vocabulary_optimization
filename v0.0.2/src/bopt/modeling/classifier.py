import code
from dataclasses import dataclass
from typing import Union, List, Optional, Any

import torch
import torch.nn as nn

from bopt.modeling import Regularizers
from bopt.training import ClassificationSetup
from bopt.unigram_lm_tokenizers.tokenizers import UnigramLMTokenizerOutput

@dataclass
class ClassifierOutput:

    task_loss:Optional[Any] = None
    regularizers:Optional[Any] = None
    logits:Optional[Any] = None
    labels:Optional[Any] = None
    predictions:Optional[Any] = None

class Classifier(nn.Module):

    def __init__(self, model, input_tokenizer, label_tokenizer):
        super().__init__()
        self.model = model
        self.input_tokenizer = input_tokenizer
        self.label_tokenizer = label_tokenizer

    def forward(self,
                setup: ClassificationSetup,
                ids: List[str],
                sentences: Union[List[str],
                List[List[str]]], labels: List[List[str]],
                mode):
        if mode == "train":
            tokenization_memoizer = setup.train_tokenization_memoizer
            label_memoizer = setup.train_label_memoizer
        if mode == "dev":
            tokenization_memoizer = setup.dev_tokenization_memoizer
            label_memoizer = setup.dev_label_memoizer
        if mode == "test":
            tokenization_memoizer = setup.test_tokenization_memoizer
            label_memoizer = setup.test_label_memoizer
        if isinstance(sentences[0], list):
            # multiple sentence setup
            sentence_ids = [[f"{id}-{j}" for j in range(len(sentence_group))] for id,sentence_group in zip(ids, sentences)]
        else:
            sentence_ids = ids
        label_ids = ids
        if setup.args.input_tokenizer_mode == "lattice":
            tokenizer_output: UnigramLMTokenizerOutput = self.input_tokenizer(sentences,
                                                                               setup.args.max_blocks,
                                                                               setup.args.max_unit_length,
                                                                               setup.args.max_block_length,
                                                                               setup.args.space_character,
                                                                               setup.args.split_on_space,
                                                                               setup.args.add_dummy_space_start,
                                                                               setup.args.remove_space,
                                                                               tokenization_memoizer,
                                                                               sentence_ids,
                                                                               specials=setup.specials,
                                                                               pad_token_id=self.model.config.pad_token_id)

            labels_ids= self.label_tokenizer(labels,
                                           setup.args.max_unit_length,
                                           tokenizer_output.input_ids.size(-1),
                                           label_memoizer,
                                           label_ids).to(self.input_tokenizer.device) # hack for accessing device
            losses = self.model(input_ids=tokenizer_output.input_ids,
                                position_ids=tokenizer_output.position_ids,
                                labels=labels_ids,
                                attn_bias=tokenizer_output.attention_bias)
            task_loss = losses[0]
            logits = losses[1]
            L1 = self.input_tokenizer.l1(avoid_tokens=list(setup.specials))
            shortpredictions, shortlabels = self.label_tokenizer.retrieve_predictions(self.extract_predictions(logits), labels_ids)
            return ClassifierOutput(task_loss=task_loss,
                                    regularizers=Regularizers(entropy=tokenizer_output.entropy, l1=L1, nchars=tokenizer_output.nchars),
                                    logits=logits,
                                    labels=shortlabels,
                                    predictions=shortpredictions)

    def extract_predictions(self, logits):
        return torch.argmax(logits, dim=-1)

