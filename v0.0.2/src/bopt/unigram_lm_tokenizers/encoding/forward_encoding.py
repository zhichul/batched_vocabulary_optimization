import code
from typing import List, Dict, Set
from bopt.integerize import Integerizer
import torch
import math

# all special edges have id < 0 such that we can easily filter out the true edges
NONEDGE_ID = -1
PADEDGE_ID = -2
NONEDGE_LOGPOT = -math.inf
PADEDGE_LOGPOT = 0.0

def len_c(chunk:str, specials: Set[str]):
    """
    Length of a token in characters, if it's special then it has length one.
    """
    if chunk in specials:
        return 1
    else:
        return len(chunk)

def len_block(block: List[str], specials: Set[str]):
    """
    Length of a list of tokens in total in characters.
    """
    return sum(len_c(chunk, specials) for chunk in block)
def blockify(chunks: List[str], max_blocks: int, max_block_length: int, specials=set()) -> List[List[str]]:
    """
    Given list of strings, groups them into blocks such that the total length
    of a block is at most max_block_length, and the number of blocks is padded
    to max_blocks with empty string blocks.

    Returns a list of blocks, where each block is a list of strings.

    Here's an example:
    chunks = [_hello, _I, _am, _some, _code]
    max_block_length = 6
    max_blocks = 5
    blocks =  [[_hello], [_I, _am], [_some], [_code], ['']]
    where the last block is padding with empty string.
    """
    blocks = list()
    block = []
    quota = max_block_length
    for chunk in chunks:
        length = len_c(chunk, specials)
        if length > max_block_length:
            raise ValueError(f"{max_block_length} is not enough to pack {chunk} of length {length}")
        if quota >= length:
            # keep packing in current block
            block.append(chunk)
            quota -= length
        else:
            # make a new block and pack it into
            blocks.append(block)
            block = [chunk]
            quota = max_block_length - length
    if len(block) > 0:
        # add the last one if non-empty
        blocks.append(block)
    if len(blocks) > max_blocks:
        raise ValueError(f"{max_blocks} is not enough blocks for {blocks} of count {len(blocks)}")
    return blocks + [[]] * (max_blocks - len(blocks))


def integerize_for_forward(sentences: List[str],
                           max_blocks: int,
                           max_unit_length: int,
                           max_block_length: int,
                           vocabulary: Integerizer,
                           space_character : str = "▁",
                           split_on_space : bool = True,
                           add_dummy_space_start : bool = True,
                           remove_space: bool = False,
                           memoizer = None,
                           sentence_ids = None,
                           specials=set()) -> torch.Tensor:
    """
    Given a list of sentences, this method extracts substrings from each,
    represented as a 2d matrix of vocabulary ids whose corresponding substrings
    have the same length within in each row and same start position in each
    diagonal. Here's an example (where the substring instead of vocab id
    is used for illustrative purposes):

    [[h    a     t     e   ],
     [           at    ate ],
     [           hat       ],
     [                 hate]]

    To improve computational efficiency, we limit the size of the lattice by:
        1) setting a max_unit_length
           (this will disallow extra long substrings even if they are in vocab)
        2) dividing the entire lattice into concatenation of smaller sub-lattices,
        where it is assumed no edge cross between sub-lattice boundaries. This
        reduces the depth of backprop (since the depth of the computation graph
        of forward algorithm is proportional to the number of characters in the
        lattice).

    Each sub-lattice is padded to be size M x L, with non-edges and padding
    having special ids.

    space_character is special to UnigramLM. In UnigramLM, the space char
    is usually converted to a special character as preprocessing.

    Only when split_on_space is true can max_blocks be greater than 1.
    When split_on_space is false we don't split into sub-lattices.

    Returned tensor has size BxNxMxL where B is len(sentences), N is max_blocks,
    M is max_unit_length, and L is max_block_length.

    Input sentences are assumed to contain no whitespace chars on either end,
    but we will make sure to strip it again. Space is treated as a regular char.

    remove_space will remove instead of keep space as prefix.
    """
    # first check arguments
    if len(sentences) == 0: raise ValueError("empty list of sentences received")
    if max_block_length <= 0: raise ValueError("max_block_length must be positive")
    if max_blocks <= 0: raise ValueError("max_blocks must be positive")
    if max_unit_length <= 0: raise ValueError("max_unit_length must be positive")
    if not split_on_space:
        if max_blocks != 1: raise ValueError("when split_on_space is False, "
         "we only generate single lattices without subdivision")
    if remove_space and (add_dummy_space_start or not split_on_space):
        raise ValueError("When remove_space is set to True, add_dummy_space_start must be False, and split_on_space must be true")
    if memoizer is None != sentence_ids is None: raise ValueError("memoizer and sentence_ids have to be set at the same time")

    B, N, M, L = len(sentences), max_blocks, max_unit_length, max_block_length
    # this is one of the two places where modification to the input strings is done
    sentences = [sentence.strip().replace(" ", space_character) for sentence in sentences]
    outputs = []
    for i, sentence in enumerate(sentences):
        if not memoizer or (sentence_ids[i] not in memoizer): # if not caching or caching but sentence is new
            if split_on_space:
                # if split_on_white_space is true, split the sentence, chunk it, and recurse
                chunks = sentence.split(space_character)
                # this is one of the two places where modification to the input strings is done (adding dummy space)
                chunks = [(space_character if not remove_space and add_dummy_space_start and chunks[0] not in specials else "") + chunks[0]] + [(space_character if not remove_space and chunks[0] not in specials else "") + chunk for chunk in chunks[1:]]
                blocks = blockify(chunks, N, L, specials=specials)
                block_encoding = integerize_blocks(blocks, vocabulary, max_unit_length, max_block_length, specials=specials)
            else:
                # otherwise integerize the sentence as a single block
                block_encoding = integerize_blocks([[sentence]], vocabulary, M, L, specials=specials)
            if memoizer:
                memoizer[sentence_ids[i]] = block_encoding
        else: # load from cache
            block_encoding = memoizer[sentence_ids[i]]
        outputs.append(block_encoding)
    return torch.stack(outputs, dim=0)

def integerize_blocks(blocks: List[List[str]], vocabulary: Integerizer, max_unit_length: int, max_block_length: int, specials=set()):
    """
    Returns NxMxL where N is len(blocks), and M is max unit size, and L is max unit length

    """
    M,L = max_unit_length, max_block_length
    outputs = []
    for block in blocks:
        block_length = len_block(block, specials)
        chunk_start = 0
        forward_ids = torch.ones((M, L), dtype=torch.long).fill_(NONEDGE_ID)
        forward_ids[0,block_length:] = PADEDGE_ID
        for chunk in block:
            if chunk in specials:
                start = chunk_start
                length = 1
                forward_ids[length - 1, start + length - 1] = vocabulary.index(chunk)
            else:
                for start in range(chunk_start, chunk_start + len(chunk)):  # this loop is skipped for emtpy sentences
                    for length in range(1, min(block_length - start, M) + 1):
                        if length <= 0: code.interact(local=locals())
                        unit = chunk[start - chunk_start:start - chunk_start + length]
                        if length == 1 or unit in vocabulary:
                            # do indexing of all chars and all in-vocab substrings, and only characters can be unknown
                            forward_ids[length - 1, start + length - 1] = vocabulary.index(unit, unk = length == 1)
            chunk_start += len_c(chunk, specials)
        outputs.append(forward_ids)
    return torch.stack(outputs, dim=0)

def length(forward_encodings):
    L = forward_encodings.size(-1)
    return (L - (forward_encodings == PADEDGE_ID).sum(-1).sum(-1))