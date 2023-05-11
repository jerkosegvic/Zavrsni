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
from torch.cuda.amp import autocast
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from GPT2_dataset import CBTDataset

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def train(model, tokenizer, train_dataloader, validation_dataloader, optimizer, scheduler, device, epochs, sample_every, batch_size):
    total_t0 = time.time()
    training_stats = []
    model = model.to(device)

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[1].to(device)
            b_labels = batch[1].to(device)
            b_masks = batch[2].to(device)

            model.zero_grad()        

            with autocast(dtype = torch.float16):
                try:
                    outputs = model(  b_input_ids,
                                    labels=b_labels, 
                                    attention_mask = b_masks,
                                    token_type_ids=None,
                                    )
                except:
                    print("Error at step: ", step)
                    continue
                loss = outputs[0]  
                batch_loss = loss.item()
                total_train_loss += batch_loss

            if step % sample_every == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))
                model.eval()
                sample_outputs = model.generate(
                                        bos_token_id=random.randint(1,30000),
                                        do_sample=True,   
                                        top_k=50, 
                                        max_length = 200,
                                        top_p=0.95, 
                                        num_return_sequences=1
                                    )
                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
                
                model.train()
            try:
                loss.backward()
                optimizer.step()
                scheduler.step()
            except:
                print("Error at step: ", step)
                continue
        avg_train_loss = total_train_loss / len(train_dataloader)       
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
        print("")
        print("Running Validation...")

        t0 = time.time()
        model.eval()
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader: 
            b_input_ids = batch[1].to(device)
            b_labels = batch[1].to(device)
            b_masks = batch[2].to(device)  
            with torch.no_grad():        
                outputs  = model(b_input_ids, 
    #                            token_type_ids=None, 
                                attention_mask = b_masks,
                                labels=b_labels)
                loss = outputs[0]  

            batch_loss = loss.item()
            total_eval_loss += batch_loss        

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)    

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
