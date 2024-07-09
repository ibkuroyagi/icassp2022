#!/bin/bash
# datadir="data"
# datasetsdir="~/share/share_acoustic/datasets"
datadir="data"
datasetsdir="/fsws1/share/database/AudioSet"
train_dir="${datadir}/train"
eval_dir="${datadir}/eval"

if [ ! -e "${datadir}" ]; then
    mkdir -p "${datadir}"
fi
if [ -L ${train_dir} ]; then
    echo "${train_dir} has already set."
else
    ln -s ${datasetsdir} ${train_dir}
    echo "Successfully set symlink to ${train_dir}"
fi
# if [ -L ${eval_dir} ]; then
#     echo "${eval_dir} has already set."
# else
#     ln -s ${datasetsdir}/hear2021 ${eval_dir}
#     echo "Successfully set symlink to ${eval_dir}"
# fi
