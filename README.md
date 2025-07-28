# Batched Vocabulary Optimization

Codebase for encoder and decoder models that train jointly with tokenizer parameters (a unigramLM style tokenizer that produces a weighted lattice over tokenizations).

# Batched Tokenizer
## Vectorizing Dynamic Programming over Tokenization Lattices of Different Topology
[`v0.0.2/src/bopt/unigram_lm_tokenizers`](https://github.com/zhichul/bopt/tree/a2a338b102c8ec244d516a121fdac2b2cda4bde4/v0.0.2/src/bopt/unigram_lm_tokenizers) contains vectorized code for batching lattice algoritms ([forward-backward](https://github.com/zhichul/bopt/blob/main/v0.0.2/src/bopt/unigram_lm_tokenizers/inference/forward_backward.py) algorithm in various semirings for computing token marginals as well as [entropy](https://github.com/zhichul/bopt/blob/main/v0.0.2/src/bopt/unigram_lm_tokenizers/inference/entropy.py) over the tokenization distribution efficiently). 
In particular, it supports batching over lattices of **different topology** (each sequence has a different topology because the topology depends on the word) via [padding and packing](https://github.com/zhichul/bopt/tree/main/v0.0.2/src/bopt/unigram_lm_tokenizers/encoding).
This greatly speeds up training, as naively running forward-backward for each sequence in a batch ends up taking more time than running the batch through the Transformer task model on the GPU.
Numerical instability is avoided by computing in the (log, +) semiring, and entropy computed via the expectation semiring.

# Modeling
## (Expected) Random Variable Attention to Use Probabilistic Lattice in Transformer Task Model
`v0.0.2/src/bopt/modeling` includes bert variants adapted to take weighted (probabilistic) tokenization lattice as input, using lattice marginals and conditional marginals in various ways [to gate attention](https://github.com/zhichul/bopt/blob/a2a338b102c8ec244d516a121fdac2b2cda4bde4/v0.0.2/src/bopt/modeling/modeling_bert.py#L433). 

# Training
`v0.0.2/src/bopt/training` includes training code with entropy regularization on the tokenization distribution, and a sparsity regularization on the vocabulary.

# Evaluation on Real and Synthetic Tasks
We mainly experiment on morpheme prediction tasks. They provide signal for linguistic motivated token boundaries, and let us verify how well the approach worked in propagating that signal from task loss to tokenizer parameters. We use both real data from [this paper](https://aclanthology.org/2021.acl-long.279/) as well as a synthetic dataset over nonsensical words formed by a simple concatenative morphology (optional prefix + stem + optional suffix).
