#!/usr/bin/env python3

import os
import torch

# Create the Vocabulary class that maps from strings to indices and back.
class Vocabulary:
    PAD: int = 0
    UNK: int = 1

    # Initialize the Vocabulary class by creating a list  of strings, 
    # and then a mapping from strings to indices.
    def __init__(self, strings):
        self._strings = ["[PAD]", "[UNK]"]
        self._strings.extend(strings)
        self._string_map = {string: index for index, string in enumerate(self._strings)}
    
    # Returns the length of self._strings list.
    def __len__(self):
        return len(self._strings)
    
    # Iterates over all elements in self._strings list. 
    def __iter__(self):
        return iter(self._strings)
    
    # Returns the index-th string. 
    def string(self, index):
        return self._strings[index]
    
    # Return the index of a given string.
    def index(self, string):
        return self._string_map.get(string, Vocabulary.UNK)


# Create the CBMinutesDataset class.
class CBMinutesDataset:
    # Create the Dataset class.
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data_file, train=None):
            # Load the data.
            self._data = {
                "documents": [],
                "labels": [],
            }
            for line in data_file:
                line = line.rstrip("\r\n")
                label, document = line.split("\t", maxsplit=1)

                self._data["documents"].append(document)
                self._data["labels"].append(label)

            # Create or copy the label mapping.
            if train:
                self._label_vocab = train.label_vocab
            else:
                self._label_vocab = Vocabulary(sorted(set(self._data["labels"])))

        # Property to access the internal dataset (self._data) of the instance.
        @property
        def data(self):
            return self._data
        
        # Property to access the internal labels (self._label_vocab) of the instance.
        @ property
        def label_vocab(self):
            return self._label_vocab
        
        # Returns the length of the dataset.
        def __len__(self):
            return len(self._data["labels"])
        
        # Returns the index-th item of the internal dataset. 
        def __getitem__(self, index):
            return {
                "document": self._data["documents"][index],
                "label": self._data["labels"][index],
            }
        
        # Transforms the dataset.
        def transform(self, transform):
            return CBMinutesDataset.TransformedDataset(self, transform)
        
    # Create the TransformedDataset class.
    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform):
            self._dataset = dataset
            self._transform = transform

        # Returns the length of the transformed dataset.
        def __len__(self):
            return len(self._dataset)
        
        # Returns the index-th item of the transformed dataset.
        def __getitem__(self, index):
            item = self._dataset[index]
            return self._transform(*item) if isinstance(item, tuple) else self._transform(item)
        
        # Transforms the transformed dataset.
        def transform(self, transform):
            return CBMinutesDataset.TransformedDataset(self, transform)
        
    # Creates a dataset from a given filename.
    def __init__(self, directory):
        self._directory = directory

        for dataset in ["train", "dev", "test"]:
            file_path = os.path.join(self._directory, dataset, f"{dataset}.tsv")

            with open(file_path, "r") as dataset_file:
                setattr(self, dataset, self.Dataset(dataset_file, train=getattr(self, "train", None)))

    # Evaluation infrastructure
    @staticmethod
    def evaluate(gold_dataset, predictions):
        gold_labels = gold_dataset.data["labels"]

        if len(predictions) != len(gold_labels):
            raise RuntimeError("Predicition have different size than gold data!")
        
        correct = sum(gold_labels[i] == predictions[i] for i in range(len(gold_labels)))
        return 100 * correct / len(gold_labels)
    
    @staticmethod
    def evaluate_file(gold_dataset, predictions_file):
        predictions = [line.strip("\r\n") for line in predictions_file]
        return CBMinutesDataset.evaluate(gold_dataset, predictions)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--dataset", default="test", type=str, help="Gold dataset to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = CBMinutesDataset.evaluate_file(
                getattr(CBMinutesDataset("../data"), args.dataset), predictions_file)
        print("Text classification accuracy: {:.2f}%".format(accuracy))


