#!/bin/bash

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
. ./path.sh || exit 1
. ./cmd.sh || exit 1
stage=5
stop_stage=5
datadir=data
dumpdir=dump
expdir=exp
conf=conf/tuning/EfficientNet_b0.v000.yaml
# training related
train_audioset_scp=dump/audioset/feat/balanced_train_segments/feats.scp
valid_audioset_scp=dump/audioset/feat/eval_segments/feats.scp
# inference related
resume=""
verbose=1
model=EfficientNet_b0
no=v000
mode=wave
. utils/parse_options.sh || exit 1
tag=${model}/${no}/${mode}
n_gpus=1
checkpoint="exp/${tag}/checkpoint-4000steps.pkl"
hearevaldir=../../hear-eval-kit

set -euo pipefail
# if [ "${stage}" -le -1 ] && [ "${stop_stage}" -ge -1 ]; then
#     log "Stage -1: Download pre-trained weight from PANNs repo."
#     ./local/download_weight.sh
# fi
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    log "Stage 1: Set train and eval dirs from share dir."
    ./local/set_symlink.sh
    log "Create train dir's scp files."
    ./local/data.sh
fi

datasets=(eval_segments balanced_train_segments)
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    log "Stage 2: Apply preprocess from train dir's scp files."
    for dataset in eval_segments balanced_train_segments; do
        if [ ! -e "${dumpdir}/audioset/feat/${dataset}/log" ]; then
            mkdir -p "${dumpdir}/audioset/feat/${dataset}/log"
        fi
        echo -n >"${dumpdir}/audioset/feat/${dataset}/feats.scp"
        log "Start the preprocess of ${dataset}."
        ${train_cmd} "${dumpdir}/audioset/feat/${dataset}/log/dump.log" \
        python3 -m hearline_train.bin.preprocess \
            --datadir "${datadir}/train/audios/${dataset}" \
            --dumpdir "${dumpdir}/audioset/feat/${dataset}" \
            --config "${conf}" \
            --verbose "${verbose}"
    done
    echo -n >"${dumpdir}/audioset/feat/feats.scp"
    cat "${dumpdir}/audioset/feat/balanced_train_segments/feats.scp" >>"${dumpdir}/audioset/feat/feats.scp"
    log "Successfully created dump dirs."
fi
outdir="${expdir}/${tag}"
if [ ! -e "${outdir}" ]; then
    mkdir -p "${outdir}"
fi
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    log "Stage 3: Training COLA."
    train="python3 -m hearline_train.bin.cola_train"
    log "See the log via ${outdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/train.log" \
    ${train} \
        --train_audioset_scp "${train_audioset_scp}" \
        --valid_audioset_scp "${valid_audioset_scp}" \
        --config "${conf}" \
        --outdir "${outdir}" \
        --mode "${mode}" \
        --resume "${resume}" \
        --n_gpus "${n_gpus}" \
        --verbose "${verbose}"
fi
# Evaluation code is using this repo https://github.com/hearbenchmark/hear-eval-kit
embeddings_dir=${hearevaldir}/embeddings/${tag}
if [ ! -e "${embeddings_dir}" ]; then
    mkdir -p "${embeddings_dir}"
    log "Created ${embeddings_dir}"
fi
if [ -z "${checkpoint}" ]; then
    checkpoint=${expdir}/${tag}/checkpoint-4000steps.pkl
fi
if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    log "Stage 4: create embedding verctors."
    log "checkpoint path: ${checkpoint}"
    ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/embedding.log" \
    python3 -m heareval.embeddings.runner hearline_train \
        --model ${checkpoint} \
        --tasks-dir ${hearevaldir}/tasks \
        --embeddings-dir ${embeddings_dir}
fi
if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    log "Stage 5: HEAR2021 prediction."
    log "See the results: ${embeddings_dir}"
    ${train_cmd} "${outdir}/prediction.log" \
    python3 -m heareval.predictions.runner ${embeddings_dir}/hearline_train/*
    # ${train_cmd} "${outdir}/collect_score.log" \
    # python3 local/collect_score.py --no ${no}
fi
