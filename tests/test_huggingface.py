import os

import numpy as np
import pytest
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from dvclive.data.scalar import Scalar
from dvclive.huggingface import DvcLiveCallback
from dvclive.utils import parse_scalars

# pylint: disable=redefined-outer-name, unused-argument, no-value-for-parameter

task = "cola"
metric = load_metric("glue", task)
model_checkpoint = "distilbert-base-uncased"


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["sentence"], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


@pytest.fixture
def data(tokenizer):
    train = load_dataset("glue", task, split="train[:100]")
    val = load_dataset("glue", task, split="validation[:100]")

    train = train.map(
        lambda p: preprocess_function(p, tokenizer), batched=True
    )
    val = val.map(lambda p: preprocess_function(p, tokenizer), batched=True)

    return train, val


@pytest.fixture
def model():
    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_checkpoint, num_labels=2
    )


@pytest.fixture
def args():
    return TrainingArguments(
        "test-glue",
        evaluation_strategy="epoch",
        num_train_epochs=2,
    )


def test_huggingface_integration(tmp_dir, model, args, data, tokenizer):
    trainer = Trainer(
        model,
        args,
        train_dataset=data[0],
        eval_dataset=data[1],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    callback = DvcLiveCallback()
    trainer.add_callback(callback)
    trainer.train()

    assert os.path.exists("dvclive")

    logs, _ = parse_scalars(callback.dvclive)

    assert len(logs) == 10

    scalars = os.path.join("dvclive", Scalar.subfolder)
    assert os.path.join(scalars, "eval", "matthews_correlation.tsv") in logs
    assert os.path.join(scalars, "eval", "loss.tsv") in logs
    assert os.path.join(scalars, "train", "loss.tsv") in logs
    assert len(logs[os.path.join(scalars, "epoch.tsv")]) == 3
    assert len(logs[os.path.join(scalars, "eval", "loss.tsv")]) == 2


def test_huggingface_model_file(tmp_dir, model, args, data, tokenizer, mocker):
    model_path = tmp_dir / "model_hf"
    model_save = mocker.spy(model, "save_pretrained")
    tokernizer_save = mocker.spy(tokenizer, "save_pretrained")
    trainer = Trainer(
        model,
        args,
        train_dataset=data[0],
        eval_dataset=data[1],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(DvcLiveCallback(model_file=model_path))
    trainer.train()

    assert model_path.is_dir()

    assert (model_path / "pytorch_model.bin").exists()
    assert (model_path / "config.json").exists()
    assert model_save.call_count == 2

    assert (model_path / "tokenizer.json").exists()
    assert tokernizer_save.call_count == 2
