declare -A METRIC_LABELS=(
                           ["dev_accuracy"]="dev accuracy"
                           ["dev_boundary_f1"]="dev boundary f1"
                           ["dev_boundary_precision"]="dev boundary precision"
                           ["dev_boundary_recall"]="dev boundary recall"
                           ["train_accuracy"]="train accuracy"
                           ["train.100_boundary_f1"]="train boundary f1"
                           ["train.100_boundary_precision"]="train boundary precision"
                           ["train.100_boundary_recall"]="train boundary recall"
                         )
declare -A METRIC_NAMES=(
                           ["dev_accuracy"]="accuracy"
                           ["dev_boundary_f1"]="boundary_f1"
                           ["dev_boundary_precision"]="boundary_precision"
                           ["dev_boundary_recall"]="boundary_recall"
                           ["train_accuracy"]="train_accuracy"
                           ["train.100_boundary_f1"]="train_boundary_f1"
                           ["train.100_boundary_precision"]="train_boundary_precision"
                           ["train.100_boundary_recall"]="train_boundary_recall"
                         )
declare -A METRIC_YMIN=(
                           ["dev_accuracy"]="0.0"
                           ["dev_boundary_f1"]="0.0"
                           ["dev_boundary_precision"]="0.0"
                           ["dev_boundary_recall"]="0.0"
                           ["train_accuracy"]="0.0"
                           ["train.100_boundary_f1"]="0.0"
                           ["train.100_boundary_precision"]="0.0"
                           ["train.100_boundary_recall"]="0.0"
                         )

for METRIC in ${!METRIC_LABELS[@]}
do
  python3 -m experiments.plotting.violin \
    --output violin/"${METRIC_NAMES[${METRIC}]}".png \
    --violin violin/l1=0.01.runs.json l1=0.01 data ${METRIC} teal \
    --violin violin/l1=0.1.runs.json l1=0.1 data ${METRIC} orange \
    --violin violin/l1=1.0.runs.json l1=1.0 data ${METRIC} red \
    --xmap full 11k \
    --xmap small 2k \
    --width 0.9 \
    --ylim "${METRIC_YMIN[${METRIC}]}" 1.05 \
    --xlabel "number of train.100ing words" \
    --ylabel "${METRIC_LABELS[${METRIC}]}"
done
