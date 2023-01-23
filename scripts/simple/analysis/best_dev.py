import sys

from bopt.analysis.best_dev import best_dev

if __name__ == "__main__":
    file = sys.argv[1]
    field = sys.argv[2]
    others = sys.argv[3:]
    best_metric, best_step, best_others = best_dev(file, field, *others)
    print(f"{file}: {best_step}, {field}={best_metric}, ohters: {' '.join([f'{other}={best:.2f}' for (other, best) in zip(others, best_others)])}")

