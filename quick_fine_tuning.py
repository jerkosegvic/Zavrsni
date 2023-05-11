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
from training import train

def main():
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    epochs = 1
    learning_rate = 5e-4
    warmup_steps = 1e2
    epsilon = 1e-8
    sample_every = 500
    batch_size = 2
    device = torch.device("cuda")

    torch.manual_seed(42)
    dataset = load_dataset("cbt", "CN")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='-')
    
    with open('train_dataset.pkl', 'rb') as f:
        dataset_train = pickle.load(f)

    data_valid = dataset['validation']
    dataset_valid = CBTDataset(data_valid, tokenizer)
    
    train_dataloader = DataLoader(dataset_train ,  sampler = RandomSampler(dataset_train), batch_size = batch_size )
    validation_dataloader = DataLoader(dataset_valid, sampler = SequentialSampler(dataset_valid), batch_size = batch_size )

    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
    model.cuda()

    optimizer = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = total_steps)

    train(model, tokenizer, train_dataloader, validation_dataloader, optimizer, scheduler, device, epochs, sample_every, batch_size)
    model.save_pretrained('cbt_fine_tuned/')
    
if __name__ == '__main__':
    main()
