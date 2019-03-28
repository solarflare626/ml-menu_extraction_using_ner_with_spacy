import spacy
from spacy import displacy
from collections import Counter
from pprint import pprint
nlp =spacy.load('models/menu')


def extract(text):
    doc = nlp(text)
    return doc
    


f = open("data.txt","r")
text = f.read()
# text = text.replace('\n', ' ')
# text = text.replace('  ', ' ')
# while "  " in text:
#     text = text.replace('  ', ' ')
doc = extract(text)
pprint([(X.text, X.label_) for X in doc.ents])
# pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])