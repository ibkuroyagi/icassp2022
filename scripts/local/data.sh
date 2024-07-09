#!/bin/bash
datadir=data

. utils/parse_options.sh || exit 1
set -euo pipefail
audiosetdir="${datadir}/train/audios"

for dataset in eval_segments balanced_train_segments; do
    python ./local/data.py \
        --datadir "${audiosetdir}/${dataset}" \
        --verbose 1
done
