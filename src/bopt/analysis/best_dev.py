import json

BEST = {
    "eval_avg_token": "min",
    "avg_token": "min",
    "train_loss": "min",
    "eval_zero_one_loss": "max",
    "zero_one_loss": "max",
    "eval_expected_zero_one_loss": "max",
    "expected_zero_one_loss": "max",
    "eval_log_loss": "min",
    "log_loss": "min",
}
STEP="step"


def best_dev(file, field, *others):
    best_metric = None
    best_step = None
    best_others = None
    with open(file, "rt") as f:
        for line in f:
            logline = json.loads(line)
            if best_metric is None:
                best_metric = logline[field]
                best_others = [logline[other] for other in others]
                best_step = logline[STEP]
            else:
                if (BEST[field] == "min" and logline[field] < best_metric) or (BEST[field] == "max" and logline[field] > best_metric):
                    best_metric = logline[field]
                    best_others = [logline[other] for other in others]
                    best_step = logline[STEP]
    return best_metric, best_step, best_others
