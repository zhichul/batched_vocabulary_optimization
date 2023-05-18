from matplotlib import pyplot as plt

import bopt.analysis.best_dev
from bopt.core.tokenizer import Tokenizer
from bopt.data.utils import load_vocab, load_weights
from bopt.analysis.best_dev import best_dev


HEADS = [1, 2, 4]
SIZES =  [24, 96, 192]
LAYERS = [1, 2]
exps = [6, 5]
styles = ["solid", "dashed"]
colors = ["blue", "orange"]
spec_ranges = {
    6: [0.01, 0.1, 1.0],
    5: [50, 100, 200, 400],
}
log_file_templates = {
    5: "/export/a01/artifacts/bopt/syn3/exp35/42/{}/{}/{}/{}/log.json",
    6: "/export/a01/artifacts/bopt/syn3/exp36/42/{}/{}/{}/{}/log.json",
}
effective_vocab_size = {
    5: [42, 93, 187, 384],
    6: list(reversed([223, 312, 500])),
}
GOLD_EFFECTIVE_VOCAB_SIZE = 194
STYLES = ["o", "^"]
# vocab_templates = {
#     5: "/export/a01/corpora/vopt/syn/3/full/spm-unigram-weights-{}.txt",
#     6: "/export/a01/artifacts/bopt/syn3/exp4-11/42/{}/768/checkpoint-600/learned_vocab.txt"
# }
# specials_templates = {
#     5:["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[WBD]", "[SP1]", "[SP2]", "[SP3]", "[SP4]", "[SP5]"],
#     6:["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[WBD]", "[SP1]", "[SP2]", "[SP3]", "[SP4]", "[SP5]"],
# }
# csp = {
#     5:None,
#     6:"@@"
# }
# TRAIN_FILE = "/export/a01/corpora/vopt/syn/3/full/train.csv"


# def load_input_tokenizer(input_vocab, weights, device, csp=None, specials=tuple()):
#     tokenizer = Tokenizer(vocab=input_vocab,
#                           weights=weights,
#                           log_space_parametrization=False,
#                           continuing_subword_prefix=csp,
#                           pad_token="[PAD]",
#                           max_unit_length=9,
#                           specials=list(specials),
#                           )
#     return tokenizer

fig, axes = plt.subplots(len(HEADS), len(SIZES), figsize=(9,9))
for h, HEAD in enumerate(HEADS):
    for s, SIZE in enumerate(SIZES):
        ax = axes[s, h]
        for l, (LAYER, style, STYLE) in enumerate(zip(LAYERS, styles, STYLES)):
            gold_best_acc, gold_step, gold_others = best_dev(
                "/export/a01/artifacts/bopt/syn3/exp22/exp22/42/{}/{}/{}/gold_test/log.json".format(LAYER, HEAD, SIZE),
                "eval_zero_one_loss",
                "test_zero_one_loss")
            ax.scatter([GOLD_EFFECTIVE_VOCAB_SIZE], [gold_others[0]], c="green", marker=STYLE, s=8)
            for exp, color in zip(exps, colors):
                data = []
                for SPEC, VSIZE in zip(spec_ranges[exp], effective_vocab_size[exp]):
                    best_acc, step, others = best_dev(log_file_templates[exp].format(LAYER, HEAD, SIZE, SPEC), "eval_zero_one_loss", "test_zero_one_loss")
                    data.append((VSIZE, others[0]))
                data = sorted(data, key=lambda x: x[0]) # sort by x (vocab size)
                x = [d[0] for d in data]
                y = [d[1] for d in data]
                ax.plot(x, y, color = color, linestyle=style)
                ax.set_ylim(0.4, 1.0)
        if s == 2:
            ax.set_xlabel(f"{HEAD} heads")
        if h == 0:
            ax.set_ylabel(f"{SIZE} hidden size", va='center', rotation='vertical')
        if s == 2 and h == 2:
            ax.plot([0],[0], color=colors[0], linestyle="dashed", label=f"2 layer E2E")
            ax.plot([0],[0], color=colors[0], linestyle="solid", label=f"1 layer E2E")
            ax.plot([0],[0], color=colors[1], linestyle="dashed", label=f"2 layer UnigramLM")
            ax.plot([0],[0], color=colors[1], linestyle="solid", label=f"1 layer UnigramLM")
            ax.scatter([0],[0], color="green", marker=STYLES[1], label=f"2 layer GoldMorph")
            ax.scatter([0],[0], color="green", marker=STYLES[0], label=f"1 layer GoldMorph")

            ax.legend()


fig.text(0.5, 0.04, 'Effective Vocabulary Size', ha='center')
fig.text(0.04, 0.5, 'Morpheme Prediction Accuracy', va='center', rotation='vertical')
plt.savefig("morpheme_prediction_sweep.png", dpi=600)
