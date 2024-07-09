#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import pandas as pd


parser = argparse.ArgumentParser(description="Preprocess audio files.")
parser.add_argument("--no", type=str, required=True, help="tag")
args = parser.parse_args()
base_dir = f"exp/eval/{args.no}/wave/hearline_train"
with open(
    os.path.join(base_dir, "speech_commands-v0.0.2-full", "test.predicted-scores.json")
) as f:
    speech_dict = json.load(f)
with open(
    os.path.join(
        base_dir, "dcase2016_task2-hear2021-full", "test.predicted-scores.json"
    )
) as f:
    dcase_dict = json.load(f)
with open(
    os.path.join(base_dir, "nsynth_pitch-v2.2.3-50h", "test.predicted-scores.json")
) as f:
    nsynth_dict = json.load(f)
df = pd.DataFrame(
    [
        [
            speech_dict["test"]["test_top1_acc"],
            dcase_dict["test"]["test_event_onset_200ms_fms"],
            dcase_dict["test"]["test_segment_1s_er"],
            nsynth_dict["test"]["test_pitch_acc"],
            nsynth_dict["test"]["test_chroma_acc"],
        ]
    ],
    columns=[
        "test_top1_acc",
        "test_event_onset_200ms_fms",
        "test_segment_1s_er",
        "test_pitch_acc",
        "test_chroma_acc",
    ],
)
save_path = os.path.join(base_dir, "score.csv")
df.to_csv(save_path, index=False)
print(f"Saved {save_path}.")
