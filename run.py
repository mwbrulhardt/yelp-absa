import json
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import torch
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from tqdm import tqdm
from transformers import AutoTokenizer


def create_pairs(ds):
    while True:

        item = next(ds)

        for aspect in ["food", "service", "price", "ambience", "anecdotes"]:
            yield {
                "business_id": item["business_id"],
                "user_id": item["user_id"],
                "s1": item["text"],
                "s2": aspect
            }


def chunk(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def run(config: dict):
    start = datetime.utcnow()

    batch_size = config["batch_size"]

    device = torch.device("cuda")

    model = torch.load(config["model_path"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

    print("Preprocessing...", datetime.utcnow())

    spark = SparkSession \
        .builder \
        .appName("ASBA") \
        .config("spark.driver.memory", "15g") \
        .config("spark.sql.shuffle.partitions", "300") \
        .getOrCreate()

    businesses = spark.read.json("data/yelp/business.json")
    tips = spark.read.json("data/yelp/tip.json")

    window_spec = Window.partitionBy("a.business_id")

    df =  tips.alias("a") \
      .join(businesses.alias("b"), tips.business_id == businesses.business_id, how="left") \
      .withColumn("tip_count", F.count("a.business_id").over(window_spec)) \
      .where(F.col("tip_count") >= 50) \
      .where((F.col("is_open") == 1) & (F.col("categories").contains("Restaurants"))) \
      .select("a.business_id", "user_id", "text")

    ds = df.toLocalIterator()
    ds = map(lambda x: x.asDict(), ds)
    ds = create_pairs(ds)
    ds = chunk(ds, batch_size)
    ds = map(lambda x: [item for item in x if item], ds)
    ds = map(lambda x: {k: [d[k] for d in x] for k in x[0]}, ds)

    print("Starting work...", datetime.utcnow())

    n = 5*df.count()
    total = n // batch_size + int(n % batch_size != 0)

    with open(config["path"], "w+") as fp:
        for batch in tqdm(ds, total=total):

            encoded = tokenizer(
                batch["s1"],
                batch["s2"],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                logits, = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            p = torch.softmax(logits, -1)
            y_pred = p.argmax(-1)
            y_pred = y_pred.detach().cpu().numpy()

            p = p.detach().cpu().numpy()
            w = np.array([0, -1, 0, 1, 0]).reshape(-1, 1)

            polarity = (p@w).flatten()

            result = (batch["business_id"], batch["user_id"], batch["s2"], y_pred, polarity)

            for bid, uid, aspect, label, polarity in zip(*result):
                record = {
                    "business_id": bid,
                    "user_id": uid,
                    "aspect": aspect,
                    "label": int(label),
                    "polarity": polarity
                }
                line = json.dumps(record) + "\n"
                fp.write(line)

    print("Work complete.", datetime.utcnow())
    print("Duration:", datetime.utcnow() - start)


@click.command()
@click.option("--model-id", type=str, default="distilbert-base-uncased")
@click.option("--batch-size", type=int, default=25)
def main(model_id: str, batch_size: int):

    model_name = model_id.replace("/", "-")
    config = {
        "model_id": model_id,
        "model_path": f"models/absa-{model_name}.pt",
        "batch_size": batch_size,
        "path": f"data/{model_name}/aspects.json"
    }

    run(config)


if __name__ == "__main__":
    main()
