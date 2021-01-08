import json
from datetime import datetime
from itertools import product

import click
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import torch
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import LongType
from pyspark.sql.window import Window
from tqdm import tqdm
from transformers import AutoTokenizer


def create_pairs(review: dict):

    aspects = ["food", "service", "price", "ambience", "anecdotes"]
    pairs = []

    for aspect in aspects:

        pairs += [{
            "business_id": review["business_id"],
            "user_id": review["user_id"],
            "s1": review["text"],
            "s2": aspect
        }]

    return pairs


def chunk(iterable, batch_size):
    n = len(iterable)
    for i in range(0, n, batch_size):
        yield iterable[i:min(i + batch_size, n)]


def run(config: dict):
    start = datetime.utcnow()

    device = torch.device("cuda")

    model = torch.load(config["model_path"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

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

    n = df.count()

    ds = df.toLocalIterator()
    ds = map(lambda x: x.asDict(), ds)
    ds = map(create_pairs, ds)
    ds = list(ds)
    ds = chunk(ds, 32)
    ds = map(lambda x: x[0], ds)
    ds = map(lambda x: {k: [d[k] for d in x] for k in x[0]}, ds)

    print("Starting work...", datetime.utcnow())

    total = (n // 32) if n % 32 == 0 else (n // 32) + 1

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

        y_pred = torch.softmax(logits, -1).argmax(-1)
        y_pred = y_pred.detach().cpu().numpy()

        result = (batch["business_id"], batch["user_id"], batch["s2"], y_pred)

        with open("data/analysis/aspects.json", "a+") as fp:
            for bid, uid, aspect, label in zip(*result):

                if label != 0:
                    record = {
                        "business_id": bid,
                        "user_id": uid,
                        "aspect": aspect,
                        "label": int(label)
                    }
                    line = json.dumps(record) + "\n"
                    fp.write(line)

    print("Work complete.", datetime.utcnow())
    print("Duration:", datetime.utcnow() - start)


@click.command()
@click.option("--model", type=str)
def main(model: str):
    config = {
        "model_name": model,
        "model_path": f"models/asba_{model.replace('/', '_')}.pt"
    }
    run(config)


if __name__ == "__main__":
    main()
