DOT:= .
.PHONY: all clean
all: virtualenv
virtualenv:
	export MKL_INTERFACE_LAYER=LP64 && \
    export MKL_THREADING_LAYER=GNU && \
	export CONDA_PYTHONBREAKPOINT_BACKUP=${CONDA_PYTHONBREAKPOINT_BACKUP:-pdb.set_trace} && \
	./setup_anaconda.sh venv venv 3.8
	. ./activate_python.sh; conda update -y -n base -c defaults conda
	. ./activate_python.sh; conda update -y -n base -c conda-forge conda
	git clone https://github.com/rwightman/pytorch-image-models.git
	. ./activate_python.sh; cd pytorch-image-models; pip install .
	. ./activate_python.sh; cd ../; pip install -e .
	. ./activate_python.sh; conda install -y pytorch-lightning=1.9.0 pysoundfile ipython ipykernel numpy pandas scikit-learn matplotlib librosa h5py PyYAML tensorboard black flake8 pdbpp -c conda-forge
	. ./activate_python.sh; conda install -y pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
clean:
	rm -fr venv miniconda.sh activate_python.sh pytorch-image-models
	find -iname "*.pyc" -delete