Clarity NLP Project
Political Response Clarity Classification with Longformer
Overview

This project focuses on the automatic classification of political interview responses according to their level of clarity.

The task consists of assigning one of the following labels to a question-answer pair:

Ambivalent – ambiguous response
Clear Non-Reply – evasive response
Clear Reply – direct and relevant response

The project uses the Longformer architecture to process long textual sequences while leveraging global attention mechanisms to emphasize relevant parts of the input.

Dataset

Dataset used:

ailsntua/QEvasion

Source:
https://huggingface.co/datasets/ailsntua/QEvasion

Dataset Statistics
Split	Samples
Train	3103
Validation	345
Test	308
Class Distribution (Train)
Label	Samples	Percentage
Ambivalent	1836	59.17%
Clear Non-Reply	320	10.31%
Clear Reply	947	30.52%
Project Pipeline

The complete workflow consists of:

Dataset loading
Data cleaning
Train/validation/test split
Tokenization
Construction of Global Attention Masks
Longformer fine-tuning
Validation
Test evaluation
Confusion Matrix generation
Classification Report generation
Model

Pretrained model:

allenai/longformer-base-4096
Why Longformer?

Unlike standard Transformer architectures, Longformer can efficiently process long documents through:

Local Attention
Global Attention

Maximum sequence length:

4096 tokens
Input Representation

Question-answer pairs are encoded using custom special tokens:

<QUESTION>
...
</QUESTION>

<ANSWER>
...
</ANSWER>

Example:

<QUESTION>
What is your position on climate policy?
</QUESTION>

<ANSWER>
We are currently evaluating several alternatives...
</ANSWER>
Global Attention Strategy

To improve classification performance, global attention is assigned to:

special tokens
first answer tokens

Configuration:

answer_global_tokens = 20

This allows the model to focus on the most informative parts of the response.

Training Configuration

Main hyperparameters:

learning_rate: 2e-5
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
num_train_epochs: 5
weight_decay: 0.01
warmup_steps: 100
Effective Batch Size
1 × 8 = 8

Gradient accumulation is used to simulate larger batches while keeping GPU memory consumption manageable.

Class Imbalance Handling

The dataset is highly imbalanced.

To address this issue, two techniques were employed:

WeightedRandomSampler

Oversamples minority classes during training.

Class Weights

Loss function weights:

Ambivalent       = 0.56
Clear Non-Reply  = 3.23
Clear Reply      = 1.09
Results
Validation Set
Metric	Score
Accuracy	0.684
Macro F1	0.672
Weighted F1	0.687
Test Set
Metric	Score
Accuracy	0.675
Macro F1	0.619
Weighted F1	0.683
Test Classification Report
Class	Precision	Recall	F1
Ambivalent	0.809	0.718	0.761
Clear Non-Reply	0.486	0.739	0.586
Clear Reply	0.478	0.544	0.509
Confusion Matrix
[[148  12  46]
 [  5  17   1]
 [ 30   6  43]]
Future Improvements

Possible future developments include:

Hyperparameter optimization
Data augmentation techniques
Ensemble methods
Comparison with other Transformer architectures
Instruction-tuned Large Language Models
Technologies Used
Python
PyTorch
Hugging Face Transformers
Hugging Face Datasets
Longformer
Scikit-learn
NumPy
Author

Francesco Lo Vetri

Master's Degree in Artificial Intelligence and Cybersecurity