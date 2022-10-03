
max_blocks = 10 # N: Number of words roughly in a sentence
max_block_length = 20 # L: number of characters in a block
max_unit_length = 20 # M: number of characters in a candidate unit
# max number of edges in a lattice for a block
max_block_tokens = max_block_length * (max_block_length + 1) // 2 - (max_block_length - max_unit_length) * (max_block_length - max_unit_length + 1) // 2
print(max_block_tokens)