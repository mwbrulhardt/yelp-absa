
import os
import zipfile
from pathlib import Path
from typing import List

import click
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


@click.command()
@click.option("--skip-yelp", is_flag=True)
def main(skip_yelp: bool):
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
    if not skip_yelp:
        api = KaggleApi()
        api.authenticate()

        data_path = "data/yelp"
        api.dataset_download_files(
            dataset="yelp-dataset/yelp-dataset",
            path=data_path
        )

        zip_path = os.path.join(data_path, "yelp-dataset.zip")
        with zipfile.ZipFile(zip_path, "r") as zp:
            zp.extractall(data_path)
        os.remove(zip_path)

        prefix = "yelp_academic_dataset_"
        for name in os.listdir(data_path):
            src = os.path.join(data_path, name)
            tgt = os.path.join(data_path, name.replace(prefix, ""))
            os.rename(src, tgt)

    # Models
    Path("models").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
