# Sentiment Analysis of Central Bank Minutes

This repository contains the code for creating the dataset, conducting sentiment analysis, and performing inference on unseen documents. It is part of the project from my private repository **Sentiment-Analysis-of-CB-Minutes**.

## Overview

The repository includes the following key components:

- **Dataset Creation**: The script `cbminutes_dataset.py` handles the compilation and preprocessing of the data, which is later used in the sentiment analysis.
- **Sentiment Analysis**: The `sentiment_analysis.py` script fine-tunes the RoBERTa model (Liu, 2019) for sentiment analysis of central bank minutes.
- **Inference on Unseen Documents**: The script `predict_sentiment.py` performs sentiment analysis on new, unseen documents, using the fine-tuned RoBERTa model.

## Note on Data and Requirements

The dataset and the complete set of requirements needed to run this code will be made available upon submission of my master’s thesis. Once submitted, the private repository will be published and will include all necessary details for reproducing the results.

Stay tuned!

## References

Parts of the code in this repository, including the entire `trainable_module.py` script, are sourced from the [npfl138 repository](https://github.com/ufal/npfl138).

Liu, Y. (2019). Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.
