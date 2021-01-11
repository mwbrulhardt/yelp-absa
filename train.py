
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import random_split
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW


def mcc(confusion: np.array) -> float:
    t = confusion.sum(0)
    p = confusion.sum(1)
    c = confusion.trace()
    s = confusion.sum().sum()

    num = c*s - t.dot(p)
    den = np.sqrt(s**2 - p.dot(p))*np.sqrt(s**2 - t.dot(t))

    return num / den


def load_data(path: str, model_id: str):

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    label_list = ["none", "negative", "neutral", "positive", "conflict"]
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    datasets = {
        "train": None,
        "test": None
    }

    for k in datasets:
        data = pd.read_csv(path + f"/{k}.csv")

        labels = list(data.label.apply(lambda x: label_map[x]))

        encoded = tokenizer(
            list(data.s1),
            list(data.s2),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        datasets[k] = torch.utils.data.TensorDataset(
            encoded["input_ids"],
            encoded["attention_mask"],
            torch.tensor(labels, dtype=torch.long).reshape(-1, 1)
        )

    return datasets["train"], datasets["test"]


def train_absa(config: dict, model_id: str, data_dir: str):

    model_name = model_id.replace("/", "-")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=5)
    model.train()
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"]
    )

    train_set, test_set = load_data(data_dir, model_id)

    cutoff = int(0.8 * len(train_set))
    train_subset, val_subset = random_split(train_set, [cutoff, len(train_set) - cutoff])

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True
    )

    # Train
    total = cutoff // config["batch_size"]

    for epoch in range(config["epochs"]):

        for i, (input_ids, attention_mask, labels) in tqdm(enumerate(train_loader), total=total):
            optimizer.zero_grad()

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss.backward()
            optimizer.step()

        # Validation
        running_loss = 0.0
        steps = 0
        confusion = np.zeros([5, 5])

        for i, (input_ids, attention_mask, labels) in enumerate(val_loader):
            with torch.no_grad():
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                loss, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                predicted = torch.softmax(logits, -1).argmax(-1)

                y_true = labels.flatten().cpu()
                y_pred = predicted.cpu()
                confusion += confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

                running_loss += loss.cpu().numpy()
                steps += 1

        summary = dict(
            loss=(running_loss / steps),
            accuracy=confusion.trace() / confusion.sum().sum(),
            mcc=mcc(confusion)
        )
        print(summary)

    # Test
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=int(config["batch_size"]),
        shuffle=True
    )

    test_loss = 0
    test_steps = 0
    confusion = np.zeros([5, 5])
    model.eval()

    for i, (input_ids, attention_mask, labels) in enumerate(test_loader):

        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            predicted = torch.softmax(logits, -1).argmax(-1)

            y_true = labels.flatten().cpu()
            y_pred = predicted.cpu()
            confusion += confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

            test_loss += loss.cpu().numpy()
            test_steps += 1

    torch.save(model, f"./models/absa-{model_name}.pt")
    print("Finished training.")

    print("Test Results:")
    summary = dict(
        model=model_id,
        loss=(test_loss / test_steps),
        accuracy=confusion.trace() / confusion.sum().sum(),
        mcc=mcc(confusion)
    )
    print(summary)

    labels = ["none", "negative", "neutral", "positive", "conflict"]
    confusion = pd.DataFrame(confusion, columns=labels)
    confusion.to_csv(f"data/{model_name}/confusion.csv", index=False)


@click.command()
@click.option("--model-id", type=str, default="distilbert-base-uncased")
@click.option("--epochs", type=int, default=4)
@click.option("--batch-size", type=int, default=24)
@click.option("--lr", type=float, default=2e-5)
def main(model_id: str, epochs: int, batch_size: int, lr: float):

    model_name = model_id.replace("/", "-")
    Path(f"data/{model_name}").mkdir(parents=True, exist_ok=True)

    train_absa(
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr
        },
        model_id=model_id,
        data_dir=os.path.abspath("./data/semeval2014")
    )


if __name__ == "__main__":
    main()
