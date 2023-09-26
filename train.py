import json
from nltk_utils import tokenize, stem, bag_of_words

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
    for pattern in intents["patterns"]:
        # tokenize the patterns
        w = tokenize(pattern)
        all_words.extend(w)
        
        xy.append((w, tag))
    
    