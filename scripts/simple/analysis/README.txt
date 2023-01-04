11-1-2022

Comparing three different runs:

1 epx1-12/run_fixed_point.sh is a coupled RAM where we run until the fixed point of the RAM on a particular input, and backpropogate the loss through two additional fixed point iterations.
    log: $BLU_ARTIFACTS/bopt/simple/exp1-12/42/0.0/768/fixed_point/log.json
2 exp2-12/run.sh seed 44 is a basline unigram tokenized language model
    log: $BLU_ARTIFACTS/bopt/simple/exp2-12/44/768/log.json
3 exp2-12-d/run.sh seed 44 is a baseline unigram tokenized language model but with a prediction setup similar to lattice models
    log: $BLU_ARTIFACTS/bopt/simple/exp2-12-d/44/768/log.json

Pos embeddings of 1 and 3 are one more than the pos embedding of the things they should look at. All prediction nodes have the same word type embedding.
All models are 8 layers 12 heads 768 hidden dim.

Questions:
 1) why is unigram doing slightly better on train?
 2) why is our model doing so much worse on dev?

Findings:
 1) Output lattice has low entropy at start and stays low entropy -> stuck with maximizing the most undertokenized path without considering other options

Added a d3 js script to visualize the lattice, and a suite of python scripts to plot training curves / dump & extract logs.


11-4-2022

A simple exploration of initializing to equally weighted paths by tuning the bias of the output layer.

Had a buggy version that miscounted the length of continuing subwords e.g. "@@a" as length three instead of one.
The results are in $BLU_ARTIFACTS/bopt/simple/exp1-13/42/0.0/768-buggy-init/log.json

After fixing it the model correctly intializes to uniform over paths.
The results are in $BLU_ARTIFACTS/bopt/simple/exp1-13/42/0.0/768/log.json

Findings:
 1) The lattice becomes sharp quickly at over the first 150 steps
    It is likely that with the basic MLE objective and RAM coupling, once the lattice becomes sharp it becomes "tunnel visioned" on making that particular tokenization better.

11-5-2022

Visualizes the exp1-13/exp1-14/exp1-15 lattices. (bias_renorm/ +uniform (path) lattice / ++marginal temperature)

Findings:

 1) bias-renorm Same observations as 11-4-2022 although the normalized initialization helped keep the entropy high early on.
 2) uniform lattice
