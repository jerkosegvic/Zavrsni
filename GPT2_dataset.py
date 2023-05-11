import torch
from torch.utils.data import Dataset

class CBTDataset(Dataset):

    def __init__(self, data, tokenizer, gpt2_type="gpt2", max_length=1024):
        
        
        self.tokenizer = tokenizer
        self.data = []
        self.input_ids = []
        self.attn_masks = []
        
        for entry in data:
            
            sentences = entry['sentences']
            question = entry['question']
            answer = entry['answer']
            question.replace('XXXXX' , answer)
            
            encodings_dict = tokenizer( ' '.join(sentences) + question,  truncation=True, max_length=max_length, padding="max_length")
            
            self.data.append(entry)
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (self.data[idx], self.input_ids[idx], self.attn_masks[idx])