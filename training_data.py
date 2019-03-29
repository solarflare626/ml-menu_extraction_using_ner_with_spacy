import json
import pandas as pd 
import numpy as np


LABEL = "MENU"
PRICE = "PRICE"



sample_anotation = {"annotations_and_examples":[{"content":"Homepage\nTowards Data Science\n\n    Data ScienceMachine LearningProgrammingVisualizationAIData JournalismContribute\n\nA Review of Named Entity Recognition (NER) Using Automatic Summarization of Resumes\nGo to the profile of Mohan Gupta\nMohan Gupta\nJul 10, 2018\n\nUnderstand what NER is and how it is used in the industry, various libraries for NER, code walk through of using NER for resume summarization.\n\nThis blog speaks about a field in Natural language Processing (NLP) and Information Retrieval (IR) called Named Entity Recognition and how we can apply it for automatically generating summaries of resumes by extracting only chief entities like name, education background, skills, etc.\nWhat is Named Entity Recognition?\n\nNamed-entity recognition (NER) (also known as entity identification, entity chunking and entity extraction) is a sub-task of information extraction that seeks to locate and classify named entities in text into pre-defined categories such as the names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc.\n\nNER systems have been created that use linguistic grammar-based techniques as well as statistical models such as machine learning. Hand-crafted grammar-based systems typically obtain better precision, but at the cost of lower recall and months of work by experienced computational linguists . Statistical NER systems typically require a large amount of manually annotated training data. Semi-supervised approaches have been suggested to avoid part of the annotation effort.\nState-of-the-Art NER Models\nspaCy NER Model :\n\nBeing a free and an open-source library, spaCy has made advanced Natural Language Processing (NLP) much simpler in Python.\n\nspaCy provides an exceptionally efficient statistical system for named entity recognition in python, which can assign labels to groups of tokens which are contiguous. It provides a default model which can recognize a wide range of named or numerical entities, which include company-name, location, organization, product-name, etc to name a few. Apart from these default entities, spaCy enables the addition of arbitrary classes to the entity-recognition model, by training the model to update it with newer trained examples.\n\nModel Architecture :\n\nThe statistical models in spaCy are custom-designed and provide an exceptional performance mixture of both speed, as well as accuracy. The current architecture used has not been published yet, but the following video gives an overview as to how the model works with primary focus on NER model.\nStanford Named Entity Recognizer :\n\nStanford NER is a Named Entity Recognizer, implemented in Java. It provides a default trained model for recognizing chiefly entities like Organization, Person and Location. Apart from this, various models trained for different languages and circumstances are also available.\n\nModel Architecture :\n\nStanford NER is also referred to as a CRF (Conditional Random Field) Classifier as Linear chain Conditional Random Field (CRF) sequence models have been implemented in the software. We can train our own custom models with our own labeled dataset for various applications.\n\nCRF models were originally pioneered by Lafferty, McCallum, and Pereira (2001); Please refer to Sutton and McCallum (2006) or Sutton and McCallum (2010) for detailed comprehensible introductions.\nUse Cases of NER Models\n\nNamed Entity Recognition has a wide range of applications in the field of Natural Language Processing and Information Retrieval. Few such examples have been listed below :\nAutomatically Summarizing Resumes :\n\nOne of the key challenges faced by the HR Department across companies is to evaluate a gigantic pile of resumes to shortlist candidates. To add to their burden, resumes of applicants are often excessively populated in detail, of which, most of the information is irrelevant to what the evaluator is seeking. With the aim of simplifying this process, through our NER model, we could facilitate evaluation of resumes at a quick glance, thereby simplifying the effort required in shortlisting candidates among a pile of resumes.\nOptimizing Search Engine Algorithms :\n\nTo design a search engine algorithm, instead of searching for an entered query across the millions of articles and websites online, a more efficient approach would be to run an NER model on the articles once and store the entities associated with them permanently. The key tags in the search query can then be compared with the tags associated with the website articles for a quick and efficient search.\nPowering Recommender Systems :\n\nNER can be used in developing algorithms for recommender systems which automatically filter relevant content we might be interested in and accordingly guide us to discover related and unvisited relevant contents based on our previous behaviour. This may be achieved by extracting the entities associated with the content in our history or previous activity and comparing them with label assigned to other unseen content to filter relevant ones.\nSimplifying Customer Support :\n\nNER can be used in recognizing relevant entities in customer complaints and feedback such as Product specifications, department or company branch details, so that the feedback is classified accordingly and forwarded to the appropriate department responsible for the identified product.\n\nWe describe summarization of resumes using NER models in detail in the further sections.\nNER For Resume Summarization\nDataset :\n\nThe first task at hand of course is to create manually annotated training data to train the model. For this purpose, 220 resumes were downloaded from an online jobs platform. These documents were uploaded to Dataturks online annotation tool and manually annotated.\n\nThe tool automatically parses the documents and allows for us to create annotations of important entities we are interested in and generates JSON formatted training data with each line containing the text corpus along with the annotations.\n\nA snapshot of the dataset can be seen below :\n\nThe above dataset consisting of 220 annotated resumes can be found here. We train the model with 200 resume data and test it on 20 resume data.\nUsing spaCy model in python for training a custom model :\n\nDataset format :\n\nA sample of the generated json formatted data generated by the Dataturks annotation tool, which is supplied to the code is as follows :\n\nTraining the Model :\n\nWe use python’s spaCy module for training the NER model. spaCy’s models are statistical and every “decision” they make — for example, which part-of-speech tag to assign, or whether a word is a named entity — is a prediction. This prediction is based on the examples the model has seen during training.\n\nThe model is then shown the unlabelled text and will make a prediction. Because we know the correct answer, we can give the model feedback on its prediction in the form of an error gradient of the loss function that calculates the difference between the training example and the expected output. The greater the difference, the more significant the gradient and the updates to our model.\n\nWhen training a model, we don’t just want it to memorise our examples — we want it to come up with theory that can be generalised across other examples. After all, we don’t just want the model to learn that this one instance of “Amazon” right here is a company — we want it to learn that “Amazon”, in contexts like this, is most likely a company. In order to tune the accuracy, we process our training examples in batches, and experiment with minibatch sizes and dropout rates.\n\nOf course, it’s not enough to only show a model a single example once. Especially if you only have few examples, you’ll want to train for a number of iterations. At each iteration, the training data is shuffled to ensure the model doesn’t make any generalisations based on the order of examples.\n\nAnother technique to improve the learning results is to set a dropout rate, a rate at which to randomly “drop” individual features and representations. This makes it harder for the model to memorise the training data. For example, a 0.25dropout means that each feature or internal representation has a 1/4 likelihood of being dropped. We train the model for 10 epochs and keep the dropout rate as 0.2.\n\nHere’s a code snippet for training the model :\nDataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy\n\nContribute to Entity-Recognition-In-Resumes-SpaCy development by creating an account on GitHub.\ngithub.com\n\nResults and Evaluation of the spaCy model :\n\nThe model is tested on 20 resumes and the predicted summarized resumes are stored as separate .txt files for each resume.\n\nFor each resume on which the model is tested, we calculate the accuracy score, precision, recall and f-score for each entity that the model recognizes. The values of these metrics for each entity are summed up and averaged to generate an overall score to evaluate the model on the test data consisting of 20 resumes. The entity wise evaluation results can be observed below . It is observed that the results obtained have been predicted with a commendable accuracy.\n\nA sample summary of an unseen resume of an employee from indeed.com obtained by prediction by our model is shown below :\nResume of an Accenture employee obtained from indeed.com\nSummarized Resume as obtained in output\nUsing Stanford NER model in Java for training a custom model :\n\nDataset Format :\n\nThe data for training has to be passed as a text file such that every line contains a word-label pair, where the word and the label tag are separated by a tab space ‘\\t’. For a text document,as in our case, we tokenize documents into words and add one line for each word and associated tag into the training file. To indicate the start of the next file, we add an empty line in the training file.\n\nHere is a sample of the input training file:\n\nNote: It is compulsory to include a label/tag for each word. Here, for words we do not care about we are using the label zero ‘0’.\n\nProperties file :\n\nStanford CoreNLP requires a properties file where the parameters necessary for building a custom model. For instance, we may define ways of extracting features for learning, etc. Following is an example of a properties file:\n\n# location of the training file\ntrainFile = ./standford_train.txt\n# location where you would like to save (serialize) your\n# classifier; adding .gz at the end automatically gzips the file,\n# making it smaller, and faster to load\nserializeTo = ner-model.ser.gz\n\n# structure of your training file; this tells the classifier that\n# the word is in column 0 and the correct answer is in column 1\nmap = word=0,answer=1\n\n# This specifies the order of the CRF: order 1 means that features\n# apply at most to a class pair of previous class and current class\n# or current class and next class.\nmaxLeft=1\n\n# these are the features we'd like to train with\n# some are discussed below, the rest can be\n# understood by looking at NERFeatureFactory\nuseClassFeature=true\nuseWord=true\n# word character ngrams will be included up to length 6 as prefixes\n# and suffixes only\nuseNGrams=true\nnoMidNGrams=true\nmaxNGramLeng=6\nusePrev=true\nuseNext=true\nuseDisjunctive=true\nuseSequences=true\nusePrevSequences=true\n# the last 4 properties deal with word shape features\nuseTypeSeqs=true\nuseTypeSeqs2=true\nuseTypeySequences=true\n#wordShape=chris2useLC\nwordShape=none\n#useBoundarySequences=true\n#useNeighborNGrams=true\n#useTaggySequences=true\n#printFeatures=true\n#saveFeatureIndexToDisk = true\n#useObservedSequencesOnly = true\n#useWordPairs = true\n\nTraining the model :\n\nThe chief class in Stanford CoreNLP is CRFClassifier, which possesses the actual model. In the code provided in the Github repository, the link to which has been attached below, we have provided the code to train the model using the training data and the properties file and save the model to disk to avoid time consumption for training each time. Next time we use the model for prediction on an unseen document, we just load the trained model from disk and use to for classification.\n\nThe first column in the output contains the input tokens while the second column refers to the correct label, and the third column is the label predicted by the classifier.\n\nHere’s a Code snippet for training the model and saving it to disk:\nDataTurks-Engg/Entity-Recognition-In-Resumes-StanfordNER\n\nContribute to Entity-Recognition-In-Resumes-StanfordNER development by creating an account on GitHub.\ngithub.com\n\nResults and Evaluation of the Stanford NER model :\n\nThe model is tested on 20 resumes and the predicted summarized resumes are stored as separate .txt files for each resume.\n\nFor each resume on which the model is tested, we calculate the accuracy score, precision, recall and f-score for each entity that the model recognizes. The values of these metrics for each entity are summed up and averaged to generate an overall score to evaluate the model on the test data consisting of 20 resumes. The entity wise evaluation results can be observed below . It is observed that the results obtained have been predicted with a commendable accuracy.\n\nA sample summary of an unseen resume of an employee from indeed.com obtained by prediction by our model is shown below :\nA resume of an Accenture employee obtained from indeed.com\nSummarized Resume as obtained in Output\nComparison of spaCy , Stanford NER and State-of-the-Art Models :\n\nThe vast majority of tokens in real-world resume documents are not part of entity names as usually defined, so the baseline precision, recall is extravagantly high, typically >90%; going by this logic, the entity wise precision recall values of both the models are reasonably good.\n\nFrom the evaluation of the models and the observed outputs, spaCy seems to outperform Stanford NER for the task of summarizing resumes. A review of the F-scores for the entities identified by both models is as follows :\n\nHere is the dataset of the resumes tagged with NER entities.\n\nThe Python code for the above project for training the spaCy model can be found here in the github repository.\n\nThe Java code for the above project for training the Stanford NER model can be found here in the GitHub repository.\n\nNote: This blog is an extended version of the NER blog published at Dataturks.\nThanks to Hamza Bendemra.\n\n    Machine LearningNamed Entity RecognitionNLP\n\nGo to the profile of Mohan Gupta\nMohan Gupta\nTowards Data Science\nTowards Data Science\n\nSharing concepts, ideas, and codes.\nMore from Towards Data Science\n10 Python Pandas tricks that make your work more efficient\nGo to the profile of Shiu-Tang Li\nShiu-Tang Li\nMar 13\nMore from Towards Data Science\nSix Recommendations for Aspiring Data Scientists\nGo to the profile of Ben Weber\nBen Weber\nMar 18\nMore from Towards Data Science\nA Complete Exploratory Data Analysis and Visualization for Text Data\nGo to the profile of Susan Li\nSusan Li\nMar 19\nResponses\n\nTowards Data Science\nNever miss a story from Towards Data Science, when you sign up for Medium. Learn more\n","metadata":{},"annotations":[],"classifications":[]}],"relations":{}}
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

        datas.append(sample_anotation)
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

        


