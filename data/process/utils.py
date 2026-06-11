import os
import zipfile

import certifi
import requests
from tqdm import tqdm

targetpaths = {
    "sleep_eeg": "data/sleep_eeg",
    "synthetic": "data/synthetic",
}

zippaths = {
    "sleep_eeg": "data/sleep_eeg.zip",
}

links = {
    "sleep_eeg": "https://physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip",
}

processed_path = {
    "sleep_eeg": "data/sleep_eeg/processed",
    "synthetic": "data/synthetic/processed",
}


def _ssl_verify():
    """Use certifi CA bundle; set PULSE_INSECURE_DOWNLOAD=1 to skip verify on broken clusters."""
    if os.environ.get("PULSE_INSECURE_DOWNLOAD", "").lower() in {"1", "true", "yes"}:
        return False
    return certifi.where()


def download_file(url, filename):
    """Download url to filename with streaming progress. Handles cluster SSL via certifi."""
    chunk_size = 1024
    verify = _ssl_verify()
    with requests.get(url, stream=True, verify=verify, timeout=60) as r:
        r.raise_for_status()
        total_size = int(r.headers["Content-Length"]) if r.headers.get("Content-Length") else None
        with open(filename, "wb") as f:
            pbar = tqdm(unit="B", total=total_size, unit_scale=True)
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    pbar.update(len(chunk))
                    f.write(chunk)
    return filename


def unzip_file(zippath, targetpath, remove=True):
    with zipfile.ZipFile(zippath, "r") as zip_ref:
        zip_ref.extractall(targetpath)
    if remove:
        os.remove(zippath)


def downloadextract(key, redownload=False):
    targetpath = targetpaths[key]
    zippath = zippaths[key]
    link = links[key]
    if os.path.exists(targetpath) and redownload is False:
        print(f"{key} files already exist")
        return

    print(f"Downloading {key} files ...")
    download_file(link, zippath)

    print(f"Unzipping {key} files ...")
    unzip_file(zippath, targetpath, remove=True)

    print("Done extracting and downloading")
