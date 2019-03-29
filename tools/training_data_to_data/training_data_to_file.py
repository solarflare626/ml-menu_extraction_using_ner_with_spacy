import simplejson

LABEL = "MENU"
PRICE = "PRICE"
DATA = [
            ("Whitefish Toast\n\ntwo eggs any style, crispy capers, lettuces and multigrain\n\n$18", {"entities": [(0, 15, LABEL), (77, 80, PRICE)]}),
            ("\nWhitefish Toast\n\ntwo eggs any style, crispy capers, lettuces and multigrain\n\n$18\n", {"entities": [(1, 16, LABEL), (78, 81, PRICE)]}),
            ("Spinach-Artichoke Benedict\n\npoached eggs, hollandaise, home fries\n\n$20\n", {"entities": [(0, 26, LABEL), (67, 70, PRICE)]}),
            ("\nSpinach-Artichoke Benedict\n\npoached eggs, hollandaise, home fries\n\n$20\n", {"entities": [(1, 27, LABEL), (68, 71, PRICE)]}),
            ("Bagel & Lox\n\nhouse cured gravlox, cream cheese, tomato, red onion and capers\n\n$18", {"entities": [(0, 11, LABEL), (78, 81, PRICE)]}),
            ("\nBagel & Lox\n\nhouse cured gravlox, cream cheese, tomato, red onion and capers\n\n$18\n", {"entities": [(1, 12, LABEL), (79, 82, PRICE)]}),
        ]

f = open('output.txt', 'w')
simplejson.dump(DATA, f)
f.close()

f = open('output.txt', 'r')
data =simplejson.loads(f.read())
f.close()
print(data)