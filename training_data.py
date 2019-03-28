import json
LABEL = "MENU"
PRICE = "PRICE"
class Data:
    @staticmethod
    def training_data():
        return [
            ("Whitefish Toast\n\ntwo eggs any style, crispy capers, lettuces and multigrain\n\n$18", {"entities": [(0, 15, LABEL), (77, 80, PRICE)]}),
            ("\nWhitefish Toast\n\ntwo eggs any style, crispy capers, lettuces and multigrain\n\n$18\n", {"entities": [(1, 16, LABEL), (78, 81, PRICE)]}),
            ("Spinach-Artichoke Benedict\n\npoached eggs, hollandaise, home fries\n\n$20\n", {"entities": [(0, 26, LABEL), (67, 70, PRICE)]}),
            ("\nSpinach-Artichoke Benedict\n\npoached eggs, hollandaise, home fries\n\n$20\n", {"entities": [(1, 27, LABEL), (68, 71, PRICE)]}),
            ("Bagel & Lox\n\nhouse cured gravlox, cream cheese, tomato, red onion and capers\n\n$18", {"entities": [(0, 11, LABEL), (78, 81, PRICE)]}),
            ("\nBagel & Lox\n\nhouse cured gravlox, cream cheese, tomato, red onion and capers\n\n$18\n", {"entities": [(1, 12, LABEL), (79, 82, PRICE)]}),
        ]
    @staticmethod
    def price_data():
        return [
            ("$1", {"entities": [(0, 2, PRICE)]}),
            ("$1\n", {"entities": [(0, 2, PRICE)]}),
            ("\n$1", {"entities": [(1, 3, PRICE)]}),
            ("\n$1\n", {"entities": [(1, 3, PRICE)]}),
            ("$20", {"entities": [(0, 3, PRICE)]}),
            ("$20\n", {"entities": [(0, 3, PRICE)]}),
            ("\n$20", {"entities": [(1, 4, PRICE)]}),
            ("\n$20\n", {"entities": [(1, 4, PRICE)]}),
            
            ("$1.50", {"entities": [(0, 5, PRICE)]}),
            ("$1\n", {"entities": [(0, 5, PRICE)]}),
            ("\n$1.50", {"entities": [(1, 6, PRICE)]}),
            ("\n$1.50\n", {"entities": [(1, 6, PRICE)]}),
            ("$20.99", {"entities": [(0, 6, PRICE)]}),
            ("$20.99\n", {"entities": [(0, 6, PRICE)]}),
            ("\n$20.99", {"entities": [(1, 7, PRICE)]}),
            ("\n$20.99\n", {"entities": [(1, 7, PRICE)]}),

            ("£1", {"entities": [(0, 2, PRICE)]}),
            ("£1\n", {"entities": [(0, 2, PRICE)]}),
            ("\n£1", {"entities": [(1, 3, PRICE)]}),
            ("\n£1\n", {"entities": [(1, 3, PRICE)]}),
            ("£20", {"entities": [(0, 3, PRICE)]}),
            ("£20\n", {"entities": [(0, 3, PRICE)]}),
            ("\n£20", {"entities": [(1, 4, PRICE)]}),
            ("\n£20\n", {"entities": [(1, 4, PRICE)]}),
            
            ("£1.50", {"entities": [(0, 5, PRICE)]}),
            ("£1\n", {"entities": [(0, 5, PRICE)]}),
            ("\n£1.50", {"entities": [(1, 6, PRICE)]}),
            ("\n£1.50\n", {"entities": [(1, 6, PRICE)]}),
            ("£20.99", {"entities": [(0, 6, PRICE)]}),
            ("£20.99\n", {"entities": [(0, 6, PRICE)]}),
            ("\n£20.99", {"entities": [(1, 7, PRICE)]}),
            ("\n£20.99\n", {"entities": [(1, 7, PRICE)]}),
        ]

    @staticmethod
    def lighttag_data(filePath):
        training_data = []
        print("Extracting data from lighttag datas: ", filePath)
        f = open(filePath)
        text = f.read()
        datas = json.loads(text)
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

