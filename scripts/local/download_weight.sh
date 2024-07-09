#!/bin/bash

cache_dir="exp/.cache"
target_weight="Cnn14_DecisionLevelAtt.pth"
if [ -e "${cache_dir}/${target_weight}" ]; then
    echo "${target_weight} has already saved at ${cache_dir}."
else
    echo "Start to download pre-trained weights from PANNs."
    if [ ! -d "${cache_dir}" ]; then
        mkdir -p "${cache_dir}"
    fi
    wget -O "${cache_dir}/${target_weight}" "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelAtt_mAP%3D0.425.pth"
    echo "SUccessfully downloaded ${target_weight} from PANNs."
fi
