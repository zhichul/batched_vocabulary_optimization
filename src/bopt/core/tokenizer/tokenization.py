from typing import List, Tuple, Any, Callable, TypeVar, Set, Optional

import torch

from bopt.core.integerize import Integerizer

T = TypeVar("T", str, List[str])

class TokenizationMixin:

    def __init__(self, *args,
                 vocab: Integerizer = None,
                 continuing_subword_prefix: str = None,
                 pad_token: str = "[PAD]",
                 max_unit_length: int = 1e9,
                 specials: List[str] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.parallel_backward_mask_cache = dict()
        self.vocab_list = vocab
        self.csp = continuing_subword_prefix
        self.pad_token = pad_token
        self.max_unit_length = min(max_unit_length, max(len(u) for u in vocab))

        # represent vocabulary
        self.vocab = vocab
        self.specials_set = set() if not specials else set(specials)

    @classmethod
    def len_chunk(cls, chunk: str, specials_set: Set[str]):
        return 1 if specials_set and chunk in specials_set else len(chunk)

    def len_c(self, chunk: str):
        return self.len_chunk(chunk, self.specials_set)

    def len_p(self, packed_chunk: List[str]):
        return sum(self.len_c(chunk) for chunk in packed_chunk)

    def pack_chunks(self, chunks: List[str], C: int):
        packed_chunks = list()
        packed_chunk = list()
        quota = C
        for chunk in chunks:
            length = self.len_c(chunk)
            if length > C:
                raise ValueError(f"{C} is not enough to pack {chunk} of length {length}")
            if quota >= length:
                packed_chunk.append(chunk)
                quota -= length
            else:
                packed_chunks.append(packed_chunk)
                packed_chunk = [chunk]
                quota = C - length
        if len(packed_chunk) > 0:
            packed_chunks.append(packed_chunk)
        return packed_chunks

    def integerize_packed_chunks(self, packed_chunks: List[List[str]], M:int, L:int):
        E = (L * (L+1))//2 - ((L-M) * (L-M+1)) // 2
        N = len(packed_chunks)
        ids = torch.zeros((E * N,), dtype=torch.long)
        ids.fill_(self.vocab.index(self.pad_token))
        pos_ids = torch.zeros((E * N,), dtype=torch.long)
        mask = torch.zeros((E * N,), dtype=torch.long)
        pos = 0
        for i, packed_chunk in enumerate(packed_chunks):
            j = 0
            pack_offset = 0
            for chunk in packed_chunk:
                if chunk in self.specials_set:
                    ids[i * E + pack_offset] = self.vocab.index(chunk)
                    mask[i * E + pack_offset] = 1
                    pos_ids[i * E + pack_offset] = pos
                    local_offset = min(M, L - j)
                    pos += 1
                else:
                    local_offset = 0
                    for local_offset_s, s in enumerate(range(len(chunk))):
                        for local_offset_l, l in enumerate(range(1, min(M, len(chunk) - s) + 1)):
                            unit = chunk[s:s + l]
                            unit = unit if self.csp is None or s == 0 else self.csp + unit
                            if unit in self.vocab:
                                ids[i * E + pack_offset + local_offset + local_offset_l] = self.vocab.index(unit)
                                mask[i * E + pack_offset + local_offset + local_offset_l] = 1
                                pos_ids[i * E + pack_offset + local_offset + local_offset_l] = pos
                        local_offset += min(M, L - (s + j))
                        pos += 1
                assert local_offset == sum(min(M, L - v) for v in range(j, j+self.len_c(chunk)))
                pack_offset += local_offset
                j += self.len_c(chunk)
            assert pack_offset == sum(min(M, L - v) for v in range(0, sum(self.len_c(chunk) for chunk in packed_chunk)))
        assert pos_ids.max().item() == (sum(self.len_c(chunk) for packed_chunk in packed_chunks for chunk in packed_chunk ) - 1)
        return ids, mask, pos_ids

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

    def encode_batch(self, chunks: List[str], M: int, L: int = None, device: str = "cpu") -> Tuple[torch.LongTensor,
                                                                            torch.FloatTensor,
                                                                            torch.LongTensor,
                                                                            torch.LongTensor,
                                                                            torch.FloatTensor,
                                                                            torch.LongTensor,
                                                                            torch.FloatTensor,
                                                                            torch.FloatTensor,]:
        if L is None:
            L = max(self.len_c(chunk) for chunk in chunks)  # max length of chunk
        return self.encode_batch_generic(chunks, L, M, self.encode_transitions, self.len_c, device=device)

    def encode_packed_batch(self, packed_chunks: List[List[str]], M: int, L: int = None, device: str = "cpu") -> Tuple[torch.LongTensor,
                                                                            torch.FloatTensor,
                                                                            torch.LongTensor,
                                                                            torch.LongTensor,
                                                                            torch.FloatTensor,
                                                                            torch.LongTensor,
                                                                            torch.FloatTensor,
                                                                            torch.FloatTensor,]:
        if L is None:
            L = max(self.len_p(packed_chunk) for packed_chunk in packed_chunks)  # max length of packed chunk
        return self.encode_batch_generic(packed_chunks, L, M, self.encode_packed_transitions, self.len_p, device=device)

    def encode_batch_generic(self, chunks: List[T],
                     L: int,
                     M: int,
                     encoder: Callable[[List[T], int, int], Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]],
                     length_fn: Callable[[List[T]], int],
                     device: str = "cpu") -> Tuple[torch.LongTensor,
                                                    torch.FloatTensor,
                                                    torch.LongTensor,
                                                    torch.LongTensor,
                                                    torch.FloatTensor,
                                                    torch.LongTensor,
                                                    torch.FloatTensor,
                                                    torch.FloatTensor,]:
        B = len(chunks)
        M = min(self.max_unit_length, L, M)

        # encode lattice as special transition matrices
        fwd_ids = []
        fwd_ms = []
        bwd_ids = []
        bwd_ms = []
        lengths = []
        for chunk in chunks:
            fwd_id, fwd_m, bwd_id, bwd_m = encoder(chunk, L, M, device=device)
            fwd_ids.append(fwd_id)
            fwd_ms.append(fwd_m)
            bwd_ids.append(bwd_id)
            bwd_ms.append(bwd_m)
            lengths.append(length_fn(chunk))

        # ids and ms are [B x max_length x max_length] tensors
        # lengths is a [B] tensor
        fwd_ids = torch.stack(fwd_ids)  # torch.FloatTensor
        fwd_ms = torch.stack(fwd_ms)  # torch.FloatTensor
        bwd_ids = torch.stack(bwd_ids)  # torch.FloatTensor
        bwd_ms = torch.stack(bwd_ms)  # torch.FloatTensor
        lengths = torch.tensor(lengths, dtype=torch.long,  device=device)  # torch.LongTensor

        mmask, emask = self.parallel_backward_mask(L, M,  device=device)
        mmask = mmask.unsqueeze(0)
        emask = emask.unsqueeze(0)
        bwd_ids = (bwd_ids.unsqueeze(1) * emask.to(torch.long)).reshape(B * L, M, L)
        bwd_ms = (bwd_ms.unsqueeze(1) * mmask).reshape(B * L, M, L)
        bwd_lengths = L * torch.repeat_interleave(torch.ones_like(lengths), L)
        return fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask

    @staticmethod
    def init_transitions_and_masks(M:int, L:int, device: str = "cpu"):
        fwd_mask = torch.zeros((M, L), dtype=torch.float, device=device)  # whenever a element of a matrix is from weights, set to 1
        bwd_mask = torch.zeros((M, L), dtype=torch.float, device=device)  # whenever a element of a matrix is from weights, set to 1
        fwd_ids = torch.zeros((M, L), dtype=torch.int, device=device)
        bwd_ids = torch.zeros((M, L), dtype=torch.int, device=device)
        bwd_mask[0, :] = 1 # make sure to pad the lattice for backward
        return fwd_ids, fwd_mask, bwd_ids, bwd_mask

    def encode_transitions(self, chunk: str, L: int, M: int, device: str = "cpu") -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
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
        if chunk not in self.specials_set and len(chunk) > L:
            raise ValueError(f"chunk length of {chunk} is greater than allowed max chunk length {L}")
        fwd_ids, fwd_mask, bwd_ids, bwd_mask = self.init_transitions_and_masks(M, L, device=device)

        # handle special tokens
        if chunk in self.specials_set:
            fwd_ids[0,0] = self.vocab.index(chunk)
            fwd_mask[0,0] = 1
            bwd_ids[0,0] = self.vocab.index(chunk)
            bwd_mask[0,0] = 1
            return fwd_ids, fwd_mask, bwd_ids, bwd_mask

        # handle real tokens
        for s in range(len(chunk)):
            for l in range(min(len(chunk) - s, M) + 1):
                unit = chunk[s:s + l]
                unit = unit if self.csp is None or s == 0 else self.csp + unit
                if unit in self.vocab:
                    # fwd
                    fwd_mask[l - 1, s + l - 1] = 1
                    fwd_ids[l - 1, s + l - 1] = self.vocab.index(unit)
                    # bwd
                    bwd_mask[l - 1, L - s - 1] = 1
                    bwd_ids[l - 1, L - s - 1] = self.vocab.index(unit)
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
        total_length = sum(self.len_c(chunk) for chunk in packed_chunk)
        if total_length > L:
            raise ValueError(f"chunk length of {packed_chunk} is greater than allowed max chunk length {L}")
        fwd_ids, fwd_mask, bwd_ids, bwd_mask = self.init_transitions_and_masks(M, L, device=device)

        start = 0
        for chunk in packed_chunk:
            length = self.len_c(chunk)
            cfi, cfm, cbi, cbm = self.encode_transitions(chunk, L=length, M=min(M,length), device=device)
            r, c = cfi.size()
            fwd_ids[:r, start:start + c] = cfi
            fwd_mask[:r, start:start + c] = cfm
            bwd_ids[:r, L - start - c:L - start] = cbi
            bwd_mask[:r, L - start - c:L - start] = cbm
            start += length
        return fwd_ids, fwd_mask, bwd_ids, bwd_mask
