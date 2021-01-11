
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import seaborn as sns
import torch
from pyspark.sql import SparkSession


plt.style.use("seaborn")
plt.rcParams.update({
    "figure.titlesize": 30,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.title_fontsize": 20,
    "legend.fontsize": 16
})


@click.command()
@click.option("--model-id", type=str)
def make_charts(model_id: str):
    spark = SparkSession \
        .builder \
        .appName("ASBA") \
        .config("spark.driver.memory", "15g") \
        .config("spark.sql.shuffle.partitions", "300") \
        .getOrCreate()

    businesses = spark.read.json("data/yelp/business.json")
    reviews = spark.read.json("data/yelp/review.json")
    users = spark.read.json("data/yelp/user.json")
    tips = spark.read.json("data/yelp/tip.json")

    model_name = model_id.replace('/', '-')
    path = Path(f"charts/{model_name}")
    path.mkdir(parents=True, exist_ok=True)

    # Confusion Heatmap
    confusion = pd.read_csv(f"data/{model_name}/confusion.csv")
    labels = list(confusion.columns)

    confusion = confusion.values

    fig, ax = plt.subplots(figsize=(11, 9))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    X = confusion / confusion.sum(1, keepdims=True)

    sns.heatmap(
        pd.DataFrame(X, index=labels, columns=labels),
        cmap=cmap,
        center=0,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".3f",
        linewidths=.5,
        cbar_kws={
            "shrink": .5
        }
    )
    ax.set_title(f"Confusion Matrix for {model_id} on Test Set")
    ax.set_xlabel("Truth")
    ax.set_ylabel("Predicted")
    plt.tight_layout()
    fig.savefig(path.joinpath("confusion.png"))


    # Review vs Tip Word Length
    review_wc = reviews \
        .withColumn("word_count", F.size(F.split(F.col("text"), " "))) \
        .select("word_count") \
        .toPandas()
    review_wc["type"] = "review"


    tip_wc = tips \
        .withColumn("word_count", F.size(F.split(F.col("text"), " "))) \
        .select("word_count") \
        .toPandas()
    tip_wc["type"] = "tip"

    word_count = pd.concat([review_wc, tip_wc], axis=0, ignore_index=True)

    fig, ax = plt.subplots(figsize=(15, 10))

    sns.kdeplot(
        data=word_count,
        x="word_count",
        hue="type",
        log_scale=True,
        cumulative=True,
        common_norm=False,
        common_grid=True,
        ax=ax
    )

    ax.set_title("Cumulative Dist. Comparison for Tips vs. Reviews")

    ax.set_xlabel("Number of Words")
    ax.set_ylabel("$F(x)$")

    plt.tight_layout()
    fig.savefig(path.joinpath("comparison.png"))

    del word_count, review_wc, tip_wc

    # Bias Correction Chart
    adjusted = reviews \
        .join(users, reviews.user_id == users.user_id) \
        .withColumn("adjusted_stars", (1 / 2)*(F.col("stars") - F.col("average_stars")) + 3) \
        .select("business_id", "stars", "adjusted_stars") \
        .groupBy("business_id") \
        .mean()

    ratings = adjusted.toPandas()
    ratings["bias"] = ratings["avg(stars)"] - ratings["avg(adjusted_stars)"]


    df1 = ratings.get(["business_id", "avg(stars)"])
    df1 = df1.rename({"avg(stars)": "stars"}, axis=1)
    df1["adjusted"] = False

    df2 = ratings.get(["business_id", "avg(adjusted_stars)"])
    df2 = df2.rename({"avg(adjusted_stars)": "stars"}, axis=1)
    df2["adjusted"] = True

    combined = pd.concat([df1, df2], axis=0, ignore_index=True)


    fig, ax = plt.subplots(1, 2, figsize=(25, 10))

    fig.suptitle("Review Star Ratings Bias Correction")

    sns.kdeplot(data=combined, x="stars", hue="adjusted", multiple="stack", bw_adjust=4, clip=(1, 5), ax=ax[0])
    ax[0].set_title("Distribution of Stars Comparison")
    ax[0].set_xlabel("Stars")
    ax[0].set_xlim(1, 5)

    sns.kdeplot(data=ratings, x="bias", fill=True, color="purple", ax=ax[1], clip=(-4, 4))
    ax[1].set_title("Distribution of Bias Between Stars and Adjusted Stars")
    ax[1].set_xlabel("Bias")
    ax[1].set_xlim(-4, 4)

    plt.tight_layout()
    fig.savefig(path.joinpath("bias_correction.png"))


    # Aspect Pair Plots
    aspects = spark.read.json(f"data/{model_name}/aspects.json")
    aspects = aspects.select("business_id", "user_id", "aspect", "polarity").groupBy(["business_id"]).pivot("aspect").mean()
    aspects = aspects.toPandas()
    aspects = aspects.set_index("business_id")
    aspects = 2*aspects + 3

    g = sns.pairplot(aspects, kind="hist")
    g.fig.suptitle("Aspect Relationship Pair Plots")
    g.fig.set_size_inches(15, 15)
    g.fig.savefig(path.joinpath("aspect_pairs.png"))


    # Answer Variations Chart
    bdf = businesses.alias("a") \
        .join(adjusted.alias("b"), businesses.business_id == adjusted.business_id) \
        .select("a.business_id", "name", "city", "state", "stars", "review_count", F.col("avg(adjusted_stars)").alias("adjusted_stars", )) \
        .toPandas()
    bdf = bdf.set_index("business_id")


    results = pd.concat([bdf, aspects], join="inner", axis=1)
    results = results.rename({"adjusted_stars": "overall"}, axis=1)

    features = ["overall", "food", "service", "price", "ambience", "anecdotes"]
    R = results.get(features)
    W = np.random.dirichlet([7, 10, 10, 5, 5, 5], size=[100000])

    idx = (R.values@W.T).argmax(0)

    s = results.loc[results.index[idx], :].name.value_counts()

    s = s.to_frame().reset_index()
    s.columns = ["name", "count"]
    s = s[s["count"] > 10]

    fig, ax = plt.subplots(figsize=(20, 10))

    sns.barplot(x="count", y="name", data=s, ax=ax)
    ax.set_title("Variation in Answers with Random Importance Weights")
    plt.tight_layout()
    fig.savefig(path.joinpath("variation_in_answers.png"))


if __name__ == "__main__":
    make_charts()
