# hearline-train
- python3.8
- CUDA11.3
## Install
```
git clone https://git.linecorp.com/ibuki-kuroyanagi/hearline-train.git
cd hearline-train/tools
make virtualenv
```
## Prepare Audioset
./scripts/data/train
└── audios
  ├── balanced_train_segments/*.wav
  └── eval_segments/*.wav

## Prepare evaluation code
Note : Due to the library alignment, comment out the torch and tensorflow GPU check codes in lines 21~24 of `hear-eval-kit/heareval/embeddings/runner.py`. It works without any problem.
```
git clone https://github.com/neuralaudio/hear-eval-kit
cd ~/hear-eval-kit
pip3 install -e .
```

## Experiments
```
cd scripts
./run.sh
```

## Citation
```
@INPROCEEDINGS{9746798,
  author={Kuroyanagi, Ibuki and Komatsu, Tatsuya},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Self-Supervised Learning Method Using Multiple Sampling Strategies for General-Purpose Audio Representation}, 
  year={2022},
  volume={},
  number={},
  pages={3263-3267},
  keywords={Event detection;Conferences;Signal processing;Acoustics;Task analysis;Speech processing;contrastive learning;metric learning;pitch shift;sampling strategy},
  doi={10.1109/ICASSP43922.2022.9746798}}
```
