
max_blocks = 10
max_block_length = 20 # number of characters in a block
max_unit_length = 20 # number of characters in a candidate unit
max_block_tokens = max_block_length * (max_block_length + 1) // 2 - (max_block_length - max_unit_length) * (max_block_length - max_unit_length + 1) # max number of tokens in a lattice for a block