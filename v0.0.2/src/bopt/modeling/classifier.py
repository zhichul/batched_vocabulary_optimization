import code
from dataclasses import dataclass
from typing import Union, List, Optional, Any

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

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
                                attn_bias=tokenizer_output.attention_bias,
                                token_type_ids=tokenizer_output.type_ids)
            task_loss = losses[0]
            logits = losses[1]
            L1 = self.input_tokenizer.l1(avoid_tokens=list(setup.specials))
            shortpredictions, shortlabels = self.label_tokenizer.retrieve_predictions(self.extract_predictions(logits), labels_ids)
            return ClassifierOutput(task_loss=task_loss,
                                    regularizers=Regularizers(entropy=tokenizer_output.entropy, l1=L1, nchars=tokenizer_output.nchars),
                                    logits=logits,
                                    labels=shortlabels,
                                    predictions=shortpredictions)
        if setup.args.input_tokenizer_mode == "nbest" or setup.args.input_tokenizer_mode == "1best":
            B = len(sentences)
            n = setup.args.n if setup.args.input_tokenizer_mode == "nbest" else 1
            tokenizer_output: UnigramLMTokenizerOutput = self.input_tokenizer(sentences,
                                                                              n=n,
                                                                              use_lattice_position_ids=setup.args.use_lattice_position_ids,
                                                                              max_blocks=setup.args.max_blocks,
                                                                              max_unit_length=setup.args.max_unit_length,
                                                                              max_block_length=setup.args.max_block_length,
                                                                              space_character=setup.args.space_character,
                                                                              split_on_space=setup.args.split_on_space,
                                                                              add_dummy_space_start=setup.args.add_dummy_space_start,
                                                                              remove_space=setup.args.remove_space,
                                                                              memoizer=tokenization_memoizer,
                                                                              sentence_ids=sentence_ids,
                                                                              specials=setup.specials,
                                                                              pad_token_id=self.model.config.pad_token_id,
                                                                              subsample_vocab=setup.args.subsample_vocab)
            seq_length =  tokenizer_output.input_ids.size(-1)
            labels_ids = self.label_tokenizer(labels,
                                              seq_length,
                                              label_memoizer,
                                              label_ids).to(self.input_tokenizer.device) # hack for accessing device
            losses = self.model(input_ids=tokenizer_output.input_ids.reshape(-1,seq_length),
                                position_ids=tokenizer_output.position_ids.reshape(-1,seq_length),
                                attention_mask=tokenizer_output.attention_mask.reshape(-1,seq_length),
                                token_type_ids=tokenizer_output.type_ids.reshape(-1,seq_length))
            logits = losses[0] # B x n x seq_len x |output_vocab|
            if setup.args.input_tokenizer_mode == "nbest":
                logits = (logits.reshape(B,n,seq_length,-1) * torch.softmax(tokenizer_output.weights, -1)[..., None, None]).sum(1) # weighted sum over the n best tokenizations
            else:
                logits = logits.reshape(B,seq_length,-1) # 1best mode does not use the weights
            task_loss = CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels_ids.view(-1))
            L1 = self.input_tokenizer.l1(avoid_tokens=list(setup.specials))
            shortpredictions, shortlabels = self.label_tokenizer.retrieve_predictions(self.extract_predictions(logits),
                                                                                      labels_ids)
            return ClassifierOutput(task_loss=task_loss,
                                    regularizers=Regularizers(entropy=tokenizer_output.entropy, l1=L1,
                                                              nchars=tokenizer_output.nchars),
                                    logits=logits,
                                    labels=shortlabels,
                                    predictions=shortpredictions,)



    def extract_predictions(self, logits):
        return torch.argmax(logits, dim=-1)

