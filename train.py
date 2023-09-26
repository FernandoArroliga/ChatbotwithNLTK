import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open("intents.json", "r") as f:
    intents = json.load(f)
    
all_words = []
tags = []
xy = []

# Looping the intents json file
for intent in intents["intents"]:
    # storing the tag elements
    tag = intent["tag"]
    tags.append(tag)
    
    # looping the pattern elements
    for pattern in intent["patterns"]:
        # tokenize the patterns
        w = tokenize(pattern)
        all_words.extend(w)
        
        xy.append((w, tag))
    
# Applying lower, stem and delete punctuation sign method
ignore_words = ["?", "!", ".", ",", ":"]
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Remove duplicate elements using sets
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create Bag of Words
x_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label) # CrossEntropyLoss
    
# Create the training data
x_train = np.array(x_train)
y_train = np.array(y_train)

# Create chat dataset
class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples 
    
# Defining Hyperparameters
batch_size = 8

dataset = ChatDataSet()
train_loader = DataLoader(
    dataset=dataset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=2)