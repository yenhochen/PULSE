import os
import mne
import numpy as np

from tqdm import tqdm
from utils.io import write_json
from mne.datasets.sleep_physionet.age import fetch_data as fetch_sleepeeg_healthy
from mne.datasets.sleep_physionet.temazepam import (
    fetch_data as fetch_sleepeeg_temazepam,
)
from data.process.utils import targetpaths
from data.process.utils import targetpaths, processed_path
from mne.io import read_raw_edf
from einops import rearrange

# import glob
# from mne.io import read_raw_edf
# from mne.datasets.sleep_physionet.age import fetch_data

# import utils.dhedfreader as dhedfreader
# from datetime import datetime
# from tqdm import tqdm
# import numpy.random as npr


# import requests
# import zipfile


# from utils.utils import standardize_window
# from data.process.utils import downloadextract_files


# import matplotlib.pyplot as plt


# from mne.datasets.sleep_physionet.age import fetch_data as fetch_sleepeeg_healthy


def download_SleepEEGData(redownload=True):
    # potentially faster download, but not implemented yet
    # targetpath="data/eeg/physionet-sleep-data",

    targetpath = targetpaths["sleep_eeg"]

    if os.path.exists(targetpath) and redownload == False:
        print("EEG files already exist")
        return

    # subjects = np.arange(82)
    subjects = np.arange(20)  # this is sleep-edf-20

    files = fetch_sleepeeg_healthy(
        subjects=subjects,
        path=os.path.join(targetpath, "sleep-cassette"),
        on_missing="warn",
    )

    subjects = np.arange(22)
    files = fetch_sleepeeg_temazepam(
        subjects=subjects,
        path=os.path.join(targetpath, "sleep-telemetry"),
        # on_missing="warn"
    )

    print("Done downloading SleepEEG data")


def main():
    download_SleepEEGData()  # use mne.io
    preprocess_sleepeeg_healthy()
    # preprocess_sleepeeg_temazepam()


annotation_desc_2_event_id = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
    "Sleep stage ?": 6,
    "Movement": 6,
}


# create a new event_id that unifies stages 3 and 4
event_id = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3/4": 3,
    "Sleep stage R": 4,
}

# label_names = {"W": 0,
#                "N1": 1,
#                "N2": 2,
#                "N3": 3,
#                "REM": 4}

label_names = {
    "0": "W",
    "1": "N1",
    "2": "N2",
    "3": "N3",
    "4": "REM",
    # "5": "UNKNOWN"
}


def get_split_from_ixs(data, ixs, min_len):

    fullts = np.concatenate([data["fullts"][ix][:min_len][None] for ix in ixs])
    subseq = np.concatenate([data["subseq"][ix] for ix in ixs])
    subseq_labels = np.concatenate([data["subseq_labels"][ix] for ix in ixs])
    # names = np.concatenate([data["names"][ix] for ix in ixs])
    names = np.array([data["names"][ix] for ix in ixs])

    return fullts, subseq, subseq_labels, names


def preprocess_sleepeeg_healthy(epoch_len=30, save_subdir="healthy"):
    # data = {"subseq": [],
    #         "subseq_labels": [],
    #         "fullts": [],
    #         "names": []
    #         # "fullts_labels": []
    #         }

    path = os.path.join(targetpaths["sleep_eeg"], "sleep-cassette")

    os.makedirs(os.path.join(processed_path["sleep_eeg"], save_subdir), exist_ok=True)

    # subjects = np.arange(82)
    subjects = np.arange(20)  # this is sleep-edf-20
    files = fetch_sleepeeg_healthy(
        subjects, path=path, verbose=False, on_missing="ignore"
    )

    process_files(files, epoch_len, save_subdir=save_subdir)


def preprocess_sleepeeg_temazepam(epoch_len=30, save_subdir="temazepam"):

    path = os.path.join(targetpaths["sleep_eeg"], "sleep-telemetry")

    os.makedirs(os.path.join(processed_path["sleep_eeg"], save_subdir), exist_ok=True)

    subjects = np.arange(22)
    files = fetch_sleepeeg_temazepam(
        subjects,
        path=path,
        verbose=False,
    )

    process_files(files, epoch_len, save_subdir=save_subdir)


def standardize(x):
    return (x - x.mean()) / x.std()


def process_files(files, epoch_len=30, save_subdir="healthy"):
    data = {"subseq": [], "subseq_labels": [], "fullts": [], "names": []}

    keep_cols = ["Fpz-Cz", "Pz-Oz"]
    for ix in tqdm(range(len(files))):
        name = files[ix][0].split("/")[-1].split("-")[0][:-1]
        raw = read_raw_edf(
            files[ix][0], verbose="critical", infer_types=True, preload=True
        )
        raw = raw.apply_function(standardize, picks=np.arange(7))

        ann = mne.read_annotations(files[ix][1])
        # raw.set_annotations(ann, emit_warning=False, verbose=False)

        # keep last 30-min wake events before sleep and first 30-min wake events after
        # sleep and redefine annotations on rawraw data
        ann.crop(ann[1]["onset"] - 30 * 60, ann[-2]["onset"] + 30 * 60)
        raw.set_annotations(ann, emit_warning=False, verbose="critical")

        events_train, _ = mne.events_from_annotations(
            raw,
            event_id=annotation_desc_2_event_id,
            chunk_duration=epoch_len,
            verbose=False,
        )

        tmax = 30.0 - 1.0 / raw.info["sfreq"]  # tmax in included

        epochs = mne.Epochs(
            raw=raw,
            events=events_train,
            event_id=event_id,
            tmin=0.0,
            tmax=tmax,
            baseline=None,
            verbose="critical",
            on_missing="ignore",
            picks=keep_cols,
        )

        # fullts = raw.get_data(verbose=False)[:2].T[epochs.events[0,0]:int(epochs.events[-1,0]+epoch_len*raw.info["sfreq"])]
        # fullts = raw[keep_cols][0][epochs.events[0,0]:int(epochs.events[-1,0]+epoch_len*raw.info["sfreq"])]
        fullts = raw[keep_cols][0].T[
            epochs.events[0, 0] : int(
                epochs.events[-1, 0] + epoch_len * raw.info["sfreq"]
            )
        ]
        del raw

        data["subseq"].append(
            rearrange(epochs.get_data(verbose=False), "b c t -> b t c ")
        )
        data["subseq_labels"].append(epochs.events[:, -1])
        data["fullts"].append(fullts)
        data["names"].append(name)

    # split by trial
    ixs = np.arange(len(data["subseq"]))

    np.random.seed(1234)
    ixs = np.random.permutation(ixs)
    train_ixs = ixs[: int(0.7 * len(ixs))]
    val_ixs = ixs[int(0.7 * len(ixs)) : int(0.85 * len(ixs))]
    test_ixs = ixs[int(0.85 * len(ixs)) :]
    min_len = min([len(i) for i in data["fullts"]])

    # save each split
    save_path = processed_path["sleep_eeg"]
    splits = ["train", "val", "test"]
    all_ixs = [train_ixs, val_ixs, test_ixs]

    for split, i in zip(splits, all_ixs):
        fullts, subseq, subseq_labels, names = get_split_from_ixs(data, i, min_len)

        np.save(os.path.join(save_path, save_subdir, f"{split}_data.npy"), fullts)
        np.save(os.path.join(save_path, save_subdir, f"{split}_names.npy"), names)
        np.save(
            os.path.join(save_path, save_subdir, f"{split}_data_subseq.npy"), subseq
        )
        np.save(
            os.path.join(save_path, save_subdir, f"{split}_labels_subseq.npy"),
            subseq_labels,
        )

    write_json(label_names, os.path.join(save_path, save_subdir, "label_name.json"))


if __name__ == "__main__":
    main()
