# yelp-asba
Exploring the Yelp dataset using aspect-based sentiment analysis.

First command,
```sh
python download.py
```

Second command,
```sh
python make_dataset.py
```

Third command,
```sh
python train.py --model=distilbert-base-uncased --epochs=4 --batch_size=24 --lr=5e-2
```

Fourth command,
```sh
python run_on_tips.py --model=distilbert-base-uncased
```

### References
* https://github.com/HSLCY/ABSA-BERT-pair
* https://arxiv.org/pdf/1903.09588v1.pdf
* https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools
* https://www.yelp.com/dataset/download
