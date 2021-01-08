
import os
import zipfile
from typing import List

import pandas as pd
import torch
from bs4 import BeautifulSoup
from kaggle.api.kaggle_api_extended import KaggleApi


def preprocess(bs: BeautifulSoup) -> List[dict]:

    data = []

    for s in bs.find_all("sentence"):
        sid = s.get("id")
        s1 = s.find_all("text")[0].contents[0]

        records = []

        categories = ["food", "service", "price", "ambience", "anecdotes"]

        labels = {c: "none" for c in categories}

        for ac in s.find_all("aspectcategory"):
            category = ac.get("category")
            polarity = ac.get("polarity")

            if category.startswith("anecdotes"):
                category = "anecdotes"

            labels[category] = polarity

        records = 5*[None]
        for i, (c, label) in enumerate(labels.items()):
            data += [{
                "sentence_id": sid,
                "s1": s1,
                "s2": c,
                "label": label
            }]

    return data


if __name__ == "__main__":

    # SemEval-2014
    path = "data/semeval2014"

    bs_train = BeautifulSoup(open(path + "/Restaurants_Train.xml", "r").read())
    bs_test = BeautifulSoup(open(path + "/Restaurants_Test_Gold.xml", "r").read())

    frames = {
        "train": pd.DataFrame(preprocess(bs_train)),
        "test": pd.DataFrame(preprocess(bs_test))
    }

    for k in frames:
        frames[k].to_csv(path + f"/{k}.csv", index=False)

    # Yelp
    api = KaggleApi()
    api.authenticate()

    data_path = "data/yelp"

    for name in ["business", "tip"]:

        api.dataset_download_file(
            dataset="yelp-dataset/yelp-dataset",
            file_name=f"yelp_academic_dataset_{name}.json",
            path=data_path
        )

        file_path = data_path + f"/yelp_academic_dataset_{name}.json.zip"

        with zipfile.ZipFile(file_path, "r") as zp:
            zp.extractall(data_path)

        os.remove(file_path)

        src = data_path + f"/yelp_academic_dataset_{name}.json"
        tgt = data_path + f"/{name}.json"
        os.rename(src, tgt)

    # Models
    os.mkdir("models")
