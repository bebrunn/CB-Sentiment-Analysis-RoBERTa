#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import spacy
import torch
import transformers


# Import data set and model for inference.
from cbminutes_dataset import Vocabulary, CBMinutesDataset
from sentiment_analysis import parser, Model

# Create argsparser to adjust arguments in shell.
parser.add_argument("--model_weights", default=None, type=str, help="Path to model weights.")

class UnseenMinutes:
    # Create dataset class of NewMinutes.
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, sentences, tokenizer):
            self._sentences = sentences
            self._tokenizer = tokenizer

        def  __len__(self):
            return len(self._sentences)

        def __getitem__(self, index):
            return self._sentences[index]

        def tokenize(self, batch):
            return self._tokenizer(
                batch, padding="longest", return_attention_mask=True, return_tensors="pt"
            )

    def __init__(self, directory, nlp):
        self._directory_path = Path(directory)
        self.sentence_to_document = {}
        self.documents = {}

        # Create lists of files and their texts.
        files = list(self._directory_path.glob("*.txt"))
        files.sort(key=lambda x: x.name)
        docs = [open(file_path, "r", encoding="utf-8").read() for file_path in files]

        for doc, file_path in zip(nlp.pipe(docs), files):
            # Create dataset that assigns a list of sentences to each minute.
            self.documents[file_path.name] = [sentence.text.strip() for sentence in doc.sents]


def main(args):
    # Set the random seed and number of threads.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Load the model and its weights for inference.
    backbone = transformers.AutoModel.from_pretrained(args.backbone)
    dataset = CBMinutesDataset("../data")
    model = Model(args, backbone, dataset.train)
    model.load_state_dict(torch.load(args.model_weights))
    model.eval()

    # Move model to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load spacy NLP pipeline for sentence tokenization.
    nlp = spacy.load("en_core_web_sm")

    # Load new data.
    minutes = UnseenMinutes("../data/cnb_minutes", nlp)

    # Get string map of labels.
    labels = dataset.train.label_vocab._string_map

    # Initialize tokenizer.
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.backbone)

    # Intialize dictionary to store (sentiment, hawk_pref_score) tuple.
    document_sentiments = {}

    # Loop over documents and classify their sentences.
    for doc_name, sentences in minutes.documents.items():
        sentences_dataset = minutes.Dataset(sentences, tokenizer)
        sentences_loader = torch.utils.data.DataLoader(
            sentences_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=sentences_dataset.tokenize
        )

        all_sentence_predictions = []
        for batch in sentences_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                predictions = model(input_ids, attention_mask)
                sent_class_pred = torch.argmax(predictions, axis=1)
                logits_diff = predictions[:, labels["hawkish"]] - predictions[:, labels["dovish"]]
                all_sentence_predictions.extend(zip(sent_class_pred.cpu().numpy(), logits_diff.cpu().numpy()))

        # Vote most common class in document.
        sentiment = np.bincount([element[0] for element in all_sentence_predictions]).argmax()

        # Compute hawkish preference score for each document by averaging relative logits of hawkish and dovish.
        hawk_pref_score = np.mean([element[1] for element in all_sentence_predictions])

        # Store predicted class and hawkish preference score for each document.
        document_sentiments[doc_name] = [sentiment, hawk_pref_score]
    
    # Save the predicition
    pred_directory = "../predictions"
    os.makedirs(pred_directory, exist_ok=True)
    with open(os.path.join(pred_directory, "sentiment_predictions.tsv"), "w", encoding="utf-8") as file:
        file.write("document\tsentiment\thawk_pref_score\n")        
        for doc_name, predictions in document_sentiments.items():
                file.write(f"{doc_name}\t{dataset.train.label_vocab.string(predictions[0])}\t{predictions[1]}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
