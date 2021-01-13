# Yelp: Aspect-Based Sentiment Analysis
Exploring the Yelp dataset using Aspect-Based Sentiment Analysis.

## Installation

To get started, first install all the required libraries using,
```sh
$ pip install -r requirements.txt
```

## Part 1

The following is code associated with the first article of the 2 part Medium series

### Setup
First run the setup script to make all the proper file and directories as well as
download the Yelp dataset from Kaggle.
```sh
$ python setup.py --skip-yelp
```

### Train and Evaluate
For the `model_id` parameter, you can select any model available in the `transformers` as long as it is compatible.

Then train the model by running the following script,
```sh
$ python train.py --model-id=distilbert-base-uncased --epochs=4 --batch-size=24 --lr=5e-2
```

The following are the models I've used on this task:
* `prajjwal1/bert-tiny`
* `prajjwal1/bert-small`
* `distilbert-base-uncased`
* `bert-base-uncased`

I've found `distilbert` to be the overall best suited for this task.

<img src="charts/distilbert-base-uncased/confusion.png" alt="drawing" height="600" width="750"/>


## References
* https://arxiv.org/pdf/1903.09588v1.pdf
* https://github.com/HSLCY/ABSA-BERT-pair
* https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
* https://www.yelp.com/dataset/download
