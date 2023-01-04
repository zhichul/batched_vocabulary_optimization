import sys
import json
file = sys.argv[1]
field = sys.argv[2]
BEST = {
    "avg_token": "min",
    "train_loss": "min"
}
STEP="step"

best_metric = None
best_step = None
with open(file, "rt") as f:
    for line in f:
        logline = json.loads(line)
        if best_metric is None:
            best_metric = logline[field]
            best_step = logline[STEP]
        else:
            if (BEST[field] == "min" and logline[field] < best_metric) or (BEST[field] == "max" and logline[field] > best_metric):
                best_metric = logline[field]
                best_step = logline[STEP]

print(f"{file}: {best_step}, {field}={best_metric}")

