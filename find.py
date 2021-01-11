import click
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession


def find_results(model_id: str, importances: np.array):

    model_name = model_id.replace("/", "-")

    spark = SparkSession \
        .builder \
        .appName("ASBA") \
        .config("spark.driver.memory", "15g") \
        .config("spark.sql.shuffle.partitions", "300") \
        .getOrCreate()

    businesses = spark.read.json("data/yelp/business.json")
    reviews = spark.read.json("data/yelp/review.json")
    users = spark.read.json("data/yelp/user.json")

    adjusted = reviews \
        .join(users, reviews.user_id == users.user_id) \
        .withColumn("adjusted_stars", (1 / 2)*(F.col("stars") - F.col("average_stars")) + 3) \
        .select("business_id", "stars", "adjusted_stars") \
        .groupBy("business_id") \
        .mean()

    bdf = businesses.alias("a") \
        .join(adjusted.alias("b"), businesses.business_id == adjusted.business_id) \
        .select("a.business_id", "name", "city", "state", "stars", "review_count", F.col("avg(adjusted_stars)").alias("adjusted_stars", )) \
        .toPandas()
    bdf = bdf.set_index("business_id")

    aspects = spark.read.json(f"data/{model_name}/aspects.json")
    aspects = aspects.select("business_id", "user_id", "aspect", "polarity").groupBy(["business_id"]).pivot("aspect").mean()
    aspects = aspects.toPandas()
    aspects = aspects.set_index("business_id")
    aspects = 2*aspects + 3

    results = pd.concat([bdf, aspects], join="inner", axis=1)
    results = results.rename({"adjusted_stars": "overall"}, axis=1)

    R = results.get(["overall", "food", "service", "price", "ambience", "anecdotes"])

    results["score"] = R.values@importances

    results = results.sort_values("score", ascending=False)

    print("\n")

    top = results[:5].round(3)
    
    print("Top")
    print("===")
    print(top.to_markdown())

    print("\n")

    bottom = results.sort_values(["score", "stars"])[:5].round(3)
    print("Bottom")
    print("======")
    print(bottom.to_markdown())


@click.command()
@click.option("--model-id", type=str)
@click.option("--importances", type=str)
def main(model_id: str, importances: str):

    importances = np.array([float(i) for i in importances.split(",")])
    importances = importances / importances.sum()
    find_results(model_id, importances)


if __name__ == "__main__":
    main()
