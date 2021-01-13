# Where should I eat after the pandemic?: Decision Making with Aspect-Based Sentiment Analysis Using Transformers.
Exploring the Yelp dataset using Aspect-Based Sentiment Analysis.

## Installation

To install all the required libraries run,
```sh
$ pip install -r requirements.txt
```

## Part 1

The following code is associated with the first Medium article of my 2 part series.

### Setup
Run the setup script to create all the proper files and directories. Use the `--skip-yelp` flag to skip downloading the Yelp dataset from Kaggle.
```sh
$ python setup.py --skip-yelp
```

### Train and Evaluate
For the `model_id` parameter, you can select any model from the `transformers` library that supports sentence-pair classification.

Train the model by running the following command:
```sh
$ python train.py --model-id=distilbert-base-uncased --epochs=4 --batch-size=24 --lr=5e-2
```

These are the models I've used on this task:
* `prajjwal1/bert-tiny`
* `prajjwal1/bert-small`
* `distilbert-base-uncased`
* `bert-base-uncased`

Of these models, I've found `distilbert` to be the best.

<img src="charts/distilbert-base-uncased/confusion.png" alt="drawing" height="600" width="750"/>


## Part 2

Coming soon!

## References
* https://arxiv.org/pdf/1903.09588v1.pdf
* https://github.com/HSLCY/ABSA-BERT-pair
* https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
* https://www.yelp.com/dataset/download
