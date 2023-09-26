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
    for pattern in intent["patterns"]:
        # tokenize the patterns
        w = tokenize(pattern)
        all_words.extend(w)
        
        xy.append((w, tag))
    
# Applying lower and stem method
ignore_words = ["?", "!", ".", ",", ":"]
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Remove duplicate elements using sets
all_words = sorted(set(all_words))
tags = sorted(set(tags))

