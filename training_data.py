import json
import pandas as pd 
import numpy as np
import simplejson
import random
from random import randrange

random.seed()

SPOT_NAME = "SPOT_NAME"
ADDRESS = "ADDRESS"
BOROUGH = "BOROUGH"
CITY = "CITY"
STATE = "STATE"
ZIP_CODE = "ZIP_CODE"
AREA_NAME = "AREA_NAME"
BETWEEN = "BETWEEN"
PHONE_NUMBER = "PHONE_NUMBER"
EMAIL = "EMAIL"
SPOT_DESC = "SPOT_DESC"

MENU = "MENU"
DRINK = "DRINK"
PRICE = "PRICE"
DESC = "DESC"

MENU_PLACEHOLDERS = [
    "\n{menu}\n{desc}\n{price}\n",
    "\n{menu}\n{price}\n{desc}\n",
    "\n{menu} {price}\n{desc}\n",
    "\n{menu}\n{price}\n",
    "\n{menu}\n{desc}\n",
    "\n{menu} {price}\n",
    "\n  * {menu}\n\n\n    {desc}\n\n\n    {price}\n",
    "\n{menu}\n\n\n{desc}\n",
    "\n{menu}\n\n{price}\n\n{desc}\n",
    "{menu}{price}",
    "{menu}:\n{price}"
]

class Data:
    @staticmethod
    def v1_menu_data(filePath="datas/v1/menus.json"):
        l = len(MENU_PLACEHOLDERS)
        training_datas = []
        f = open(filePath, 'r')
        data =simplejson.loads(f.read())
        f.close()

        for d in data:
            
            placeholder = MENU_PLACEHOLDERS[randrange(l)]
            if "MENU" not in d:
                d["MENU"] = ''
            if "DESC" not in d:
                d["DESC"] = ''
            if "PRICE" not in d:
                d["PRICE"] = ''
            
            text = placeholder.format(menu=d["MENU"],desc=d["DESC"],price=d["PRICE"])
            ents = []
            if("MENU" in d and d["MENU"] != '' and d["MENU"] in text ):
                ents.append((text.index(d["MENU"]),text.index(d["MENU"])+len(d["MENU"]),MENU))
            if("PRICE" in d and d["PRICE"] != '' and  d["PRICE"] in text):
                ents.append((text.index(d["PRICE"]),text.index(d["PRICE"])+len(d["PRICE"]),PRICE))
            if("DESC" in d and d["DESC"] != '' and d["DESC"] in text):
                ents.append((text.index(d["DESC"]),text.index(d["DESC"])+len(d["DESC"]),DESC))

            td = (text, {"entities": ents})
            
            training_datas.append(td)
            training_datas.append((d["MENU"], {"entities": [(0,len(d["MENU"]),MENU)]}))
        return training_datas

    @staticmethod
    def v1_training_data(filePath):
        f = open(filePath, 'r')
        data =simplejson.loads(f.read())
        f.close()
        return data
    @staticmethod
    def read_training_data(filePath):
        f = open(filePath, 'r')
        data =simplejson.loads(f.read())
        f.close()
        return data
    
    @staticmethod
    def training_data(filePath = "datas/TRAINING_DATA.txt"):
        return Data.read_training_data(filePath)
    @staticmethod
    def price_data(filePath = "datas/PRICE_DATA.txt"):
        return Data.read_training_data(filePath)

    @staticmethod
    def lighttag_data(filePath='data/lighttag.json'):
        training_data = []
        print("Extracting data from lighttag datas: ", filePath)
        f = open(filePath)
        text = f.read()
        datas = json.loads(text)
        f.close()

        f = open("datas/ANOTATIONS.json")
        text = f.read()
        sample_anotation = json.loads(text)
        f.close()

        datas += sample_anotation
        #transform
        
        for data in datas:
            values = data["annotations_and_examples"]
            for value in values:
                content = value["content"]
                annotations = value["annotations"]
                entities = []

                for annotation in annotations:
                    entity = (annotation["start"],annotation["end"],annotation["tag"])
                    entities.append(entity)
                
                hold = (content,{"entities": entities})

                training_data.append(hold)

        return training_data

    @staticmethod
    def pizza_data():
        training_data = []
        placeholders = ["{value}","{value}\n","\n{value}","\n{value}\n"]
        
        d = pd.read_csv("datas/pizza.csv")
        pizzas = d["0"].values

        total_pizzas = len(pizzas)
        print("total pizza names: ", total_pizzas)
        print("Generating pizza training data....")
        
        for pizza in pizzas:
            for placeholder in placeholders:
                s = placeholder.format(value=pizza)
                data = (s, {"entities": [(s.index(pizza), len(pizza), MENU)]})
                training_data.append(data)

        return training_data

        


