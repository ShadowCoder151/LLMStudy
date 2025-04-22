import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import csv


# nltk.download('stopwords')
# nltk.download('punkt')

stop = set(stopwords.words('english'))
stemmer = PorterStemmer()

def whitespace_tokenizer(sentence):
    return sentence.split()

def punct_aware_tokenizer(sentence):
    return [t for t in re.split(r'(\W)', sentence) if t.strip()]

def lower_case_tokenizer(sentence):
    sentence = sentence.lower()
    return sentence.split()

def stopword_tokenizer(sentence):
    tokens = re.findall(r'\b\w+\b', sentence.lower())
    return [w for w in tokens if w not in stop]

def stem_tokenizer(sentence):
    sentence = sentence.lower()
    t1 = [t for t in re.split(r'(\W)', sentence) if t.strip()]
    return [stemmer.stem(word) for word in t1]

S = []
with open('data\\eng_sentences.tsv', 'r', encoding='utf-8') as file:
    lines = list(csv.reader(file, delimiter='\t'))
    i = 0
    while i < 5:
        S.append(lines[i][2])
        i += 1

wt = whitespace_tokenizer
pat = punct_aware_tokenizer
lct = lower_case_tokenizer
st = stopword_tokenizer
stt = stem_tokenizer

def display_results(S, function):
    for s in S:
        print(function(s))

display_results(S, stt)
