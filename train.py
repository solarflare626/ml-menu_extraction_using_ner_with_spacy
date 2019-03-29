#!/usr/bin/env python
# coding: utf8
"""Example of training an additional entity type

This script shows how to add a new entity type to an existing pre-trained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.

After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+
Last tested with: v2.1.0
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from training_data import Data
from spacy.util import minibatch, compounding
from datetime import datetime

#use gpu if cuda is available
spacy.prefer_gpu()
# new entity label
LABEL = "MENU"
PRICE = "PRICE"

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting


BAD_DATA = [
        ("Do they taste good?", {"entities": []}),
        ("gluten free", {"entities": []}),
]

MENU_ONLY_DATA = [
    ("Fried Chicken", {"entities": [(0, 13, LABEL)]}),
    ("Fried Chicken ", {"entities": [(0, 13, LABEL)]}),
    ("fried chicken ", {"entities": [(0, 13, LABEL)]}),
    ("Falafel", {"entities": [(0,7, LABEL)]}),
    ("falafel", {"entities": [(0,7, LABEL)]}),
    ("Kibbeh Bi Laban", {"entities": [(0,15, LABEL)]}),
    ("Beef Shawarma", {"entities": [(0,len("Beef Shawarma"), LABEL)]}),
    ("Beef Fried Kibbeh", {"entities": [(0,len("Beef Fried Kibbeh"), LABEL)]}),
    ("Beef Fried Kibbeh\n", {"entities": [(0,len("Beef Fried Kibbeh"), LABEL)]}),
    ("Arnabeet Mekle", {"entities": [(0,len("Arnabeet Mekle"), LABEL)]}),
    ("Pear Almond Cake", {"entities": [(0,len("Pear Almond Cake"), LABEL)]}),
    ("Chocolate Cake", {"entities": [(0,len("Chocolate Cake"), LABEL)]}),
    ("Chocolate Cake\n", {"entities": [(0,len("Chocolate Cake"), LABEL)]}),
    ("\nChocolate Cake", {"entities": [(1,len("Chocolate Cake")+1, LABEL)]}),
    ("\nChocolate Cake\n", {"entities": [(1,len("Chocolate Cake")+1, LABEL)]}),
]

GOOD_DATA = [ 
        ("Fried Chicken is good", {"entities": [(0, 13, LABEL)]}),
        
        ("Fried Chicken are too tall and they pretend to care about your feelings", {"entities": [(0, 13, LABEL)]}),
        ("Fried Chicken: fried chickpea & fava bean croquettes", {"entities": [(0, 13, LABEL)]}),
        ("a good fried chicken  contains fava bean croquettes", {"entities": [(7, 20, LABEL)]}),
        ("fried fhicken ingredients fried chickpea & fava bean croquettes", {"entities": [(0,13, LABEL)]}),
        ("Falafel fried chickpea & fava bean croquettes", {"entities": [(0,7, LABEL)]}),        
        ("falafel fried chickpea & fava bean croquettes", {"entities": [(0,7, LABEL)]}),
        ("falafel are too tall and they pretend to care about your feelings",{"entities": [(0, 7, LABEL)]}),
        ("What is falafel?",{"entities": [(8, 15, LABEL)]}),
        ("a good falafel  contains fava bean croquettes", {"entities": [(7, 14, LABEL)]},),
        ("Chicken Livers pomegranate molasses, lemon, sumac $17",{"entities": [(0, 14, LABEL )]}),
        ("Kibbeh Bi Laban beef dumplings, yogurt, kouzbara, aleppo pepper, mint $24",{"entities": [(0, 15, LABEL )]}),

]

TRAIN_DATA = GOOD_DATA + MENU_ONLY_DATA + BAD_DATA + Data.training_data() + Data.price_data() +Data.lighttag_data('data/lighttag.json')



@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("100", "option", "n", int),
)
def main(model=None, new_model_name="menu", output_dir="models/menu", n_iter=30):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(LABEL)  # add new entity label to entity recognizer
    ner.add_label(PRICE)  # add new entity label to entity recognizer
    # Adding extraneous labels shouldn't mess anything up
    # ner.add_label("VEGETABLE")
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_text = "Kibbeh Naye Beirutieh"
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":
    tstart = datetime.now()
    print("Start Time: ", tstart)

    plac.call(main)
    
    tend = datetime.now()
    print("End Time: ", tend)
    print("Total Training Time: ", tend - tstart)