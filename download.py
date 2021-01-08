
import os


os.system("kaggle datasets download -d yelp-dataset/yelp-dataset")
os.system("unzip yelp-dataset.zip -d data/yelp-dataset")
os.rename("data/yelp-dataset", "data/yelp")

for name in ["business", "checkin", "review", "tip", "user"]:
    src = f"data/yelp/yelp_academic_dataset_{name}.json"
    tgt = f"data/yelp/{name}.json"
    os.rename(src, tgt)

os.remove("yelp-dataset.zip")
