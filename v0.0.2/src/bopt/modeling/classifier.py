import code
from dataclasses import dataclass
from typing import Union, List, Optional, Any

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from bopt.bilevel import ClassificationBilevelTrainingSetup
from bopt.modeling import Regularizers
from bopt.training import ClassificationTrainingSetup
from bopt.unigram_lm_tokenizers.tokenizers import UnigramLMTokenizerOutput

@dataclass
class ClassifierOutput:

    task_loss:Optional[Any] = None
    regularizers:Optional[Any] = None
    logits:Optional[Any] = None
    labels:Optional[Any] = None
    predictions:Optional[Any] = None
    attentions:Optional[Any] = None
    input_ids:Optional[Any] = None
    position_ids:Optional[Any] = None
    type_ids:Optional[Any] = None
    attention_mask:Optional[Any] = None
    attention_bias:Optional[Any] = None
    edge_log_potentials:Optional[Any] = None
    forward_encodings:Optional[Any] = None

class Classifier(nn.Module):

    def __init__(self, model, input_tokenizer, label_tokenizer):
        super().__init__()
        self.model = model
        self.input_tokenizer = input_tokenizer
        self.label_tokenizer = label_tokenizer
        self._tokenizer_parameter_mask = tuple(name.startswith("input_tokenizer") for name, param in self.named_parameters())

    def forward(self,
                setup: Union[ClassificationTrainingSetup, ClassificationBilevelTrainingSetup],
                ids: List[str],
                sentences: Union[List[str],
                List[List[str]]], labels: List[List[str]],
                mode,
                output_attentions=False,
                output_inputs=False):
        if setup.args.gold_percentage is not None:
            references = [pair[1] for pair in sentences]
            sentences = [pair[0] for pair in sentences]
        if mode == "train":
            tokenization_memoizer = setup.train_tokenization_memoizer
            label_memoizer = setup.train_label_memoizer
        elif mode == "train_inner":
            tokenization_memoizer = setup.train_inner_tokenization_memoizer
            label_memoizer = setup.train_inner_label_memoizer
        elif mode == "train_outer":
            tokenization_memoizer = setup.train_outer_tokenization_memoizer
            label_memoizer = setup.train_outer_label_memoizer
        elif mode == "dev":
            tokenization_memoizer = setup.dev_tokenization_memoizer
            label_memoizer = setup.dev_label_memoizer
        elif mode == "test":
            tokenization_memoizer = setup.test_tokenization_memoizer
            label_memoizer = setup.test_label_memoizer
        else:
            tokenization_memoizer = None
            label_memoizer = None
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
                                                                               try_word_initial_when_unk=setup.args.try_word_initial_when_unk,
                                                                               pad_token_id=self.model.config.pad_token_id,
                                                                               temperature=setup.args.temperature,
                                                                               collapse_padding=setup.args.collapse_padding,
                                                                               output_inputs=output_inputs)

            labels_ids= self.label_tokenizer(labels,
                                           setup.args.max_unit_length,
                                           tokenizer_output.input_ids.size(-1),
                                           label_memoizer,
                                           label_ids).to(self.input_tokenizer.device) # hack for accessing device
            losses = self.model(input_ids=tokenizer_output.input_ids,
                                position_ids=tokenizer_output.position_ids,
                                labels=labels_ids,
                                attn_bias=tokenizer_output.attention_bias,
                                token_type_ids=tokenizer_output.type_ids,
                                return_dict=True,
                                output_attentions=output_attentions)
            task_loss = losses[0]
            logits = losses[1]
            L1 = self.input_tokenizer.l1(avoid_tokens=list(setup.specials))
            shortpredictions, shortlabels = self.label_tokenizer.retrieve_predictions(self.extract_predictions(logits), labels_ids)
            return ClassifierOutput(task_loss=task_loss,
                                    regularizers=Regularizers(entropy=tokenizer_output.entropy, l1=L1, nchars=tokenizer_output.nchars),
                                    logits=logits,
                                    labels=shortlabels,
                                    predictions=shortpredictions,
                                    attentions=losses["attentions"] if output_attentions else None,
                                    input_ids=tokenizer_output.input_ids if output_inputs else None,
                                    position_ids=tokenizer_output.position_ids if output_inputs else None,
                                    type_ids=tokenizer_output.type_ids if output_inputs else None,
                                    attention_mask=tokenizer_output.attention_mask if output_inputs else None,
                                    attention_bias=tokenizer_output.attention_bias if output_inputs else None,
                                    edge_log_potentials=tokenizer_output.edge_log_potentials if output_inputs else None,
                                    forward_encodings=tokenizer_output.forward_encodings if output_inputs else None)

        if setup.args.input_tokenizer_mode == "nbest" or setup.args.input_tokenizer_mode == "1best":
            B = len(sentences)
            n = setup.args.n if setup.args.input_tokenizer_mode == "nbest" and self.training else 1
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
                                                                              try_word_initial_when_unk=setup.args.try_word_initial_when_unk,
                                                                              pad_token_id=self.model.config.pad_token_id,
                                                                              subsample_vocab=setup.args.subsample_vocab,
                                                                              temperature=setup.args.temperature,
                                                                              output_inputs=output_inputs,
                                                                              max_tokens=-1 if (setup.args.gold_percentage is None or all(ref is None for ref in references)) else max(len(ref) for ref in references if ref is not None))
            seq_length =  tokenizer_output.input_ids.size(-1)
            labels_ids = self.label_tokenizer(labels,
                                              seq_length,
                                              label_memoizer,
                                              label_ids).to(self.input_tokenizer.device) # hack for accessing device

            # do gold override
            input_ids = tokenizer_output.input_ids.reshape(-1, seq_length)
            attention_mask = tokenizer_output.attention_mask.reshape(-1, seq_length)
            if setup.args.gold_percentage is not None:
                if not setup.args.input_tokenizer_mode == "1best": raise AssertionError
                for i, reference in enumerate(references):
                    if reference is not None:
                        input_ids[i, :] = 0
                        input_ids[i, :len(reference)] = torch.tensor(reference, device=input_ids.device, dtype=input_ids.dtype)
                        attention_mask[i, :] = 0
                        attention_mask[i, :len(reference)] = 1

            losses = self.model(input_ids=input_ids,
                                position_ids=tokenizer_output.position_ids.reshape(-1,seq_length),
                                attention_mask=attention_mask,
                                token_type_ids=tokenizer_output.type_ids.reshape(-1,seq_length),
                                return_dict=True,
                                output_attentions=output_attentions)
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
                                    predictions=shortpredictions,
                                    attentions=losses["attentions"] if output_attentions else None,
                                    input_ids=input_ids if output_inputs else None,
                                    position_ids=tokenizer_output.position_ids if output_inputs else None,
                                    type_ids=tokenizer_output.type_ids if output_inputs else None,
                                    attention_mask=attention_mask if output_inputs else None,
                                    attention_bias=tokenizer_output.attention_bias if output_inputs else None,
                                    edge_log_potentials=tokenizer_output.edge_log_potentials if output_inputs else None,
                                    forward_encodings=tokenizer_output.forward_encodings if output_inputs else None)
        if setup.args.input_tokenizer_mode == "bert":
            B = len(sentences)
            tokenizer_output = self.input_tokenizer.encode_batch(sentences)
            max_length = max(len(out.tokens) for out in tokenizer_output)
            if setup.args.gold_percentage is not None:
                max_length = max(max_length, -1 if all(ref is None for ref in references) else max(len(ref) for ref in references if ref is not None))
            input_ids = torch.zeros(B, max_length, dtype=torch.long)
            position_ids = torch.zeros(B, max_length, dtype=torch.long)
            attention_mask = torch.zeros(B, max_length, dtype=torch.long)
            type_ids = torch.zeros(B, max_length, dtype=torch.long)
            for i in range(input_ids.size(0)):
                l = len(tokenizer_output[i].tokens)
                input_ids[i, :l] = torch.tensor(tokenizer_output[i].ids, dtype=torch.long)
                position_ids[i, :l] = torch.tensor(list(range(l)), dtype=torch.long)
                attention_mask[i, :l] = torch.tensor(tokenizer_output[i].attention_mask, dtype=torch.long)
                type_ids[i, :l] = torch.tensor(tokenizer_output[i].type_ids, dtype=torch.long)
            input_ids = input_ids.to(setup.args.device)
            position_ids = position_ids.to(setup.args.device)
            attention_mask = attention_mask.to(setup.args.device)
            type_ids = type_ids.to(setup.args.device)
            labels_ids = self.label_tokenizer(labels,
                                              max_length,
                                              label_memoizer,
                                              label_ids).to(setup.args.device) # hack for accessing device

            # do gold override
            if setup.args.gold_percentage is not None:
                for i, reference in enumerate(references):
                    if reference is not None:
                        input_ids[i, :] = 0
                        input_ids[i, :len(reference)] = torch.tensor(reference, device=input_ids.device, dtype=input_ids.dtype)
                        attention_mask[i, :] = 0
                        attention_mask[i, :len(reference)] = 1

            losses = self.model(input_ids=input_ids,
                                position_ids=position_ids,
                                attention_mask=attention_mask,
                                token_type_ids=type_ids,
                                return_dict=True,
                                output_attentions=output_attentions)
            logits = losses[0] # B x n x seq_len x |output_vocab|
            logits = logits.reshape(B,max_length,-1) # 1best mode does not use the weights
            task_loss = CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels_ids.view(-1))
            shortpredictions, shortlabels = self.label_tokenizer.retrieve_predictions(self.extract_predictions(logits),
                                                                                      labels_ids)
            return ClassifierOutput(task_loss=task_loss,
                                    regularizers=Regularizers(entropy=None, l1=None,
                                                              nchars=None),
                                    logits=logits,
                                    labels=shortlabels,
                                    predictions=shortpredictions,
                                    attentions=losses["attentions"] if output_attentions else None,
                                    input_ids=input_ids if output_inputs else None,
                                    position_ids=position_ids if output_inputs else None,
                                    attention_mask=attention_mask if output_inputs else None)


    def extract_predictions(self, logits):
        return torch.argmax(logits, dim=-1)

    @property
    def tokenizer_parameter_mask(self):
        return self._tokenizer_parameter_mask
