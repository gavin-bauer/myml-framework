# -*- coding: utf-8 -*-
import os
import requests
import tarfile
import pandas as pd

FILE_NAMES = ["mnist_train_100.csv", "mnist_test_10.csv"]
DATASET_URLS = ["https://drive.google.com/uc?export=download&id=0B6e9Zx7axvo-SllZLVRJeWYwMzA",
                "https://drive.google.com/uc?export=download&id=0B6e9Zx7axvo-ZHJyVlhEVHB3RlE"
               ]
DATASET_PATH_ROOT = os.path.join("data", "raw")

def download_dataset(file_name, dataset_url):
    """Download a dataset from a url 
    and save it in destination folder."""
    destination = os.path.join(DATASET_PATH_ROOT, file_name)
    r = requests.get(dataset_url, stream = True)
    print(f"The dataset {file_name} was successfully downloaded.")
    with open(destination, "wb") as csv:
        for chunk in r.iter_content():
            if chunk:
                csv.write(chunk)

    if destination[-4:] == ".tgz":
        tar_file = tarfile.open(destination, "r")
        tar_file.extractall(path=os.path.join("data", "raw"))
        tar_file.close()
        os.remove(destination)

    print(f"The dataset {file_name} was successfully saved (in ../data/raw).")

if __name__ == "__main__":
    for i in range(len(FILE_NAMES)):
        download_dataset(FILE_NAMES[i], DATASET_URLS[i])
    