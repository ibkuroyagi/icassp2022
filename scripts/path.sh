# cuda related
export CUDA_HOME=/usr/local/cuda-11.3
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# path related
export PRJ_ROOT="${PWD}/.."
. "${PRJ_ROOT}/tools/activate_python.sh"
export PATH="${PATH}:utils"
export PATH="${CUDA_HOME}/bin:${PATH}"
# python related
export OMP_NUM_THREADS=1
export PYTHONIOENCODING=UTF-8
export MPL_BACKEND=Agg
