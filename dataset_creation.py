import os
import time
import datetime
import numpy as np 
import torch 
import datasets
import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from GPT2_dataset import CBTDataset

def main():

    dataset = load_dataset("cbt", "CN")
    max_length = 1024
    batch_size = 8
    data_train = dataset['train']
    data_valid = dataset['validation']
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='-')

    dataset_train = CBTDataset(data_train, tokenizer, max_length=max_length)

    file_name = 'train_dataset.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(dataset_train, file)
        print(f'Object successfully saved to "{file_name}"')


if __name__ == '__main__':
    main()