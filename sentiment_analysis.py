#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np

import torch
import torchmetrics
import transformers

from trainable_module import TrainableModule
from cbminutes_dataset import CBMinutesDataset

# Create argsparser to adjust arguments in shell.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size used for training.")
parser.add_argument("--epochs", default=10, type=int, help="Number of training epochs.")
parser.add_argument("--seed", default=17, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--backbone", default="roberta-large", type=str, help="Pre-trained transformer.")
parser.add_argument("--learning_rate", default=3e-05, type=float, help="Learning rate.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")
parser.add_argument("--save_weights", default=False, type=bool, help="Save model weights.")

# Create the Model class.
class Model(TrainableModule):
    def __init__(self, args, backbone, dataset):
        super().__init__()
        # Initialize Model class by defining it.
        self._backbone = backbone
        self._classifier = torch.nn.Linear(backbone.config.hidden_size, len(dataset.label_vocab))

    # Implement the forward pass.
    def forward(self, input_ids, attention_mask):
        hidden = self._backbone(input_ids, attention_mask).last_hidden_state
        hidden = hidden[:, 0]
        hidden = self._classifier(hidden)
        return hidden

def main(args):
    # Set the random seed and number of threads.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Creates a directory path for logging purposes. 
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the transformer model and its tokenizer.
    backbone = transformers.AutoModel.from_pretrained(args.backbone)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.backbone)

    # Load the data.
    minutes = CBMinutesDataset("../data")

    # Define functions to prepare data and create the dataloader.
    def prepare_example(example):
        return example["document"], minutes.train.label_vocab.index(example["label"])
    
    def prepare_batch(data):
        documents, labels = zip(*data)
        tokenized = tokenizer(documents, padding="longest", return_tensors="pt")
        return (tokenized.input_ids, tokenized.attention_mask), torch.tensor(labels)
    
    def create_dataloader(dataset, shuffle):
        return torch.utils.data.DataLoader(
            dataset.transform(prepare_example), args.batch_size, shuffle, collate_fn=prepare_batch
        )
    
    # Create dataloader objects from datasets
    train = create_dataloader(minutes.train, shuffle=True)
    dev = create_dataloader(minutes.dev, shuffle=False)
    test = create_dataloader(minutes.test, shuffle=False)
    
    # Create the model.
    model = Model(args, backbone, minutes.train)

    # Choose the optimizer.
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Create the number of total and warm-up steps.
    total_steps = len(train) * args.epochs
    warmup_steps = int(0.06 * total_steps)

    # Create learning rate schedule.
    schedule = transformers.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
   
    # Configure model.
    model.configure(
        optimizer=optimizer,
        schedule=schedule,
        loss=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
        metrics=torchmetrics.Accuracy(
            task="multiclass", num_classes=len(minutes.train.label_vocab)
            ),
        logdir=args.logdir,
    )

    # Train the model and evaluate it on the dev set.
    model.fit(train, dev=dev, epochs=args.epochs)

    # Save the model weights
    if args.save_weights:
        os.makedirs(args.logdir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.logdir, "model_weights.pth"))
        

    # Generate test set annotations, but in 'args.logdir' to allow for parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        # Predict the tags on the test set.
        predictions = model.predict(test)
        for sentence in predictions:
            print(minutes.train.label_vocab.string(np.argmax(sentence)), file=predictions_file)
    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)