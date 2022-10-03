from typing import List, Tuple

import torch

from bopt.core.integerize import Integerizer


class TokenizationMixin:

    def __init__(self, *args,
                 vocab: List[str] = None,
                 continuing_subword_prefix: str = None,
                 pad_token: str = "[PAD]",
                 max_unit_length: int = 1e9,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.parallel_backward_mask_cache = dict()
        self.vocab_list = vocab
        self.csp = continuing_subword_prefix
        self.pad_token = pad_token
        self.max_unit_length = min(max_unit_length, max(len(u) for u in vocab))

        # represent vocabulary
        self.vocab = Integerizer(vocab)

    def parallel_backward_mask(self, L: int, M: int, device: str = "cpu") -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if (L, M, device) in self.parallel_backward_mask_cache:
            return self.parallel_backward_mask_cache[(L, M, device)]
        mmask = torch.zeros(L, L, L, dtype=torch.float).to(device)
        emask = torch.zeros(L, L, L, dtype=torch.float).to(device)
        triu_ones = torch.triu(torch.ones(L, L, dtype=torch.float).to(device), diagonal=0)
        for j in range(L):
            # the i=2 backward pass should have a transition mask looking like this
            #   valid edges                   mmask            emask
            # [ e t  a   h   ]             [ 1 1 1 1 ]       [ 0 0 1 1 ]
            # [      at      ]             [ 0 0 0 1 ]       [ 0 0 0 1 ]
            # [      ate hat ]             [ 0 0 0 0 ]       [ 0 0 0 0 ]
            # [          hate]             [ 0 0 0 0 ]       [ 0 0 0 0 ]
            #
            # mmask has the diagonal and a single path leading up to the i=2th node (this represents the actual lattice)
            # emask only cuts out the edges themselves that's actually in the sub-lattice (this is for picking out weigths)
            i = L - 1 - j  # reverse the order so we do backward of the smallest lattice first instead of full lattice first
            mmask[j, :L - i, i:L] = triu_ones[:L - i, :L - i]
            emask[j, :L - i, i:L] = triu_ones[:L - i, :L - i]
            mmask[j, 0, :i] = 1
        mmask, emask = mmask[:,:M,:], emask[:,:M,:]
        self.parallel_backward_mask_cache[(L, M, device)] = (mmask, emask)
        return mmask, emask

    def encode_batch(self, chunks: List[str], M: int, device: str = "cpu") -> Tuple[torch.LongTensor,
                                                                            torch.FloatTensor,
                                                                            torch.LongTensor,
                                                                            torch.LongTensor,
                                                                            torch.FloatTensor,
                                                                            torch.LongTensor,
                                                                            torch.FloatTensor,
                                                                            torch.FloatTensor,]:
        L = max(len(chunk) for chunk in chunks)  # max length of chunk
        B = len(chunks)
        M = min(self.max_unit_length, L, M)

        # encode lattice as special transition matrices
        fwd_ids = []
        fwd_ms = []
        bwd_ids = []
        bwd_ms = []
        lengths = []
        for chunk in chunks:
            fwd_id, fwd_m, bwd_id, bwd_m = self.encode_transitions(chunk, L, M, device=device)
            fwd_ids.append(fwd_id)
            fwd_ms.append(fwd_m)
            bwd_ids.append(bwd_id)
            bwd_ms.append(bwd_m)
            lengths.append(len(chunk))

        # ts and ms are [B x max_length x max_length] tensors
        # lengths is a [B] tensor
        fwd_ids = torch.stack(fwd_ids)  # torch.FloatTensor
        fwd_ms = torch.stack(fwd_ms)  # torch.FloatTensor
        bwd_ids = torch.stack(bwd_ids)  # torch.FloatTensor
        bwd_ms = torch.stack(bwd_ms)  # torch.FloatTensor
        lengths = torch.tensor(lengths, dtype=torch.long,  device=device)  # torch.LongTensor

        mmask, emask = self.parallel_backward_mask(L, M,  device=device)
        mmask = mmask.unsqueeze(0)
        emask = emask.unsqueeze(0)
        bwd_ts = (bwd_ids.unsqueeze(1) * emask.to(torch.long)).reshape(B * L, M, L)
        bwd_ms = (bwd_ms.unsqueeze(1) * mmask).reshape(B * L, M, L)
        bwd_lengths = torch.repeat_interleave(lengths, L)

        return fwd_ids, fwd_ms, lengths, bwd_ts, bwd_ms, bwd_lengths, mmask, emask

    def encode_packed_batch(self, packed_chunks: List[List[str]], M: int, device: str = "cpu") -> Tuple[torch.LongTensor,
                                                                            torch.FloatTensor,
                                                                            torch.LongTensor,
                                                                            torch.LongTensor,
                                                                            torch.FloatTensor,
                                                                            torch.LongTensor,
                                                                            torch.FloatTensor,
                                                                            torch.FloatTensor,]:
        L = max(sum(len(chunk) for chunk in packed_chunk) for packed_chunk in packed_chunks)  # max length of chunk
        B = len(packed_chunks)
        M = min(self.max_unit_length, L, M)

        # encode lattice as special transition matrices
        fwd_ids = []
        fwd_ms = []
        bwd_ids = []
        bwd_ms = []
        lengths = []
        for packed_chunk in packed_chunks:
            fwd_id, fwd_m, bwd_id, bwd_m = self.encode_packed_transitions(packed_chunk, L, M, device=device)
            fwd_ids.append(fwd_id)
            fwd_ms.append(fwd_m)
            bwd_ids.append(bwd_id)
            bwd_ms.append(bwd_m)
            lengths.append(sum(len(chunk) for chunk in packed_chunk))

        # ts and ms are [B x max_length x max_length] tensors
        # lengths is a [B] tensor
        fwd_ids = torch.stack(fwd_ids)  # torch.FloatTensor
        fwd_ms = torch.stack(fwd_ms)  # torch.FloatTensor
        bwd_ids = torch.stack(bwd_ids)  # torch.FloatTensor
        bwd_ms = torch.stack(bwd_ms)  # torch.FloatTensor
        lengths = torch.tensor(lengths, dtype=torch.long,  device=device)  # torch.LongTensor

        mmask, emask = self.parallel_backward_mask(L, M,  device=device)
        mmask = mmask.unsqueeze(0)
        emask = emask.unsqueeze(0)
        bwd_ts = (bwd_ids.unsqueeze(1) * emask.to(torch.long)).reshape(B * L, M, L)
        bwd_ms = (bwd_ms.unsqueeze(1) * mmask).reshape(B * L, M, L)
        bwd_lengths = torch.repeat_interleave(lengths, L)
        return fwd_ids, fwd_ms, lengths, bwd_ts, bwd_ms, bwd_lengths, mmask, emask

    def encode_transitions(self, chunk: str, L: int, M: int, device="cpu") -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        """
        forward encoding
        [ h a  t   e   ]
        [   ha at  te  ]
        [      hat ate ]
        [          hate]

        backward encoding
        [ h   a   t  e]
        [ ha  at  te  ]
        [ hat ate     ]
        [ hate        ]
        then flip ^ left to right
        """
        if len(chunk) > L:
            raise ValueError(f"chunk length of {chunk} is greater than allowed max chunk length {L}")
        fwd_mask = torch.zeros((M, L), dtype=torch.float, device=device)  # whenever a element of a matrix is from weights, set to 1
        bwd_mask = torch.zeros((M, L), dtype=torch.float, device=device)  # whenever a element of a matrix is from weights, set to 1
        fwd_ids = torch.zeros((M, L), dtype=torch.int, device=device)
        bwd_ids = torch.zeros((M, L), dtype=torch.int, device=device)
        for s in range(len(chunk)):
            for l in range(min(len(chunk) - s, M) + 1):
                unit = chunk[s:s + l]
                unit = unit if self.csp is None or s == 0 else self.csp + unit
                if unit in self.vocab:
                    # fwd
                    fwd_mask[l - 1, s + l - 1] = 1
                    fwd_ids[l - 1, s + l - 1] = self.vocab.index(unit)
                    # bwd
                    bwd_mask[l - 1, len(chunk) - s - 1] = 1
                    bwd_ids[l - 1, len(chunk) - s - 1] = self.vocab.index(unit)
        return fwd_ids, fwd_mask, bwd_ids, bwd_mask

    def encode_packed_transitions(self, packed_chunk: List[str], L: int, M: int, device="cpu") -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        """
        forward encoding
        [ h a  t   e   ]
        [   ha at  te  ]
        [      hat ate ]
        [          hate]

        backward encoding
        [ h   a   t  e]
        [ ha  at  te  ]
        [ hat ate     ]
        [ hate        ]
        then flip ^ left to right
        """
        total_length = sum(len(chunk) for chunk in packed_chunk)
        if total_length > L:
            raise ValueError(f"chunk length of {packed_chunk} is greater than allowed max chunk length {L}")
        fwd_mask = torch.zeros((M, L), dtype=torch.float, device=device)  # whenever a element of a matrix is from weights, set to 1
        bwd_mask = torch.zeros((M, L), dtype=torch.float, device=device)  # whenever a element of a matrix is from weights, set to 1
        fwd_ids = torch.zeros((M, L), dtype=torch.int, device=device)
        bwd_ids = torch.zeros((M, L), dtype=torch.int, device=device)
        start = 0
        for chunk in packed_chunk:
            cfi, cfm, cbi, cbm = self.encode_transitions(chunk, L=len(chunk), M=min(M, len(chunk)), device=device)
            r, c = cfi.size()
            fwd_ids[:r, start:start + c] = cfi
            fwd_mask[:r, start:start + c] = cfm
            bwd_ids[:r, total_length - start - c:total_length - start] = cbi
            bwd_mask[:r, total_length - start - c:total_length - start] = cbm
            start += len(chunk)
        return fwd_ids, fwd_mask, bwd_ids, bwd_mask
