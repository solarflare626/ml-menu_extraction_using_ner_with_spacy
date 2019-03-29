import json
import pandas as pd 
import numpy as np
import simplejson

LABEL = "MENU"
PRICE = "PRICE"

class Data:
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
                data = (s, {"entities": [(s.index(pizza), len(pizza), LABEL)]})
                training_data.append(data)

        return training_data

        


