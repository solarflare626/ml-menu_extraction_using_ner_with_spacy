import spacy
from spacy import displacy
from collections import Counter
from pprint import pprint
nlp =spacy.load('models/menu')


def extract(text):
    doc = nlp(text)
    return doc
    


text = '''

Hindbeh

dandelion greens, pine nuts, caramelized onions

$14

gluten free, vegan

Riz

Lebanese rice, toasted vermicelli, mixed nuts

$10

vegan

Arnabeet Mekle

cauliflower, tahini labne, chili, mint

$16

gluten free, vegetarian

Moujadara Croquette

green lentil, caramelized onion, turmeric, yogurt

$16

gluten free, vegetarian

Rkaykat bil Jibneh

cheese rolls, kashkaval, feta, fresh mint

$13

vegetarian

Beef Fried Kibbeh

spiced beef, pine nuts, onion, yogurt

$17

Mekanek

sautéed lamb sausage, lemon, pine nuts

$17

gluten free

Kebab Kerez

lamb & beef meatballs, cherry sauce, kataifi, scallions

$18

Atayef & Veal Bacon

fig jam, pickles, hot peppers

$21

Duck Shawarma

duck magret & chicken, fig jam, green onion, garlic whip

$22

Beef Shawarma

beef & lamb, tomato, sumac onion, tahini

$21

Black Iron Shrimp

jalapeño, garlic, cilantro

$18

gluten free

Octopus

squash, onion, peppers, mekanek, pine nuts

$24

gluten free

Chicken Livers

pomegranate molasses, lemon, sumac

$17

gluten free

Kibbeh Bi Laban

beef dumplings, yogurt, kouzbara, aleppo pepper, mint

$24
'''
doc = extract(text)
pprint([(X.text, X.label_) for X in doc.ents])
# pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])