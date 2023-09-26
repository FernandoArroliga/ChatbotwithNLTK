# Importing the useful library
import nltk

# Download nltk package for make tokenize
nltk.download("punkt")

# importing the package for make stemming
from nltk.stem.porter import PorterStemmer

# Create the stemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    pass

