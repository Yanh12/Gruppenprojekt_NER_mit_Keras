# coding: utf-8
import numpy as np
def extract_features():
    def extract_sentences(path):
        """
        Extracts sentences from given text file based on CoNLL 2003 format
        :param path:
        :return:
        """
        with open(path, encoding="utf8") as file:
            sentences = []
            sentence = []
            for line in file:
                if line in ['\n', '\r\n'] or len(line) == 0 or line.startswith('-DOCSTART'):
                    if len(sentence) > 0:
                        sentences.append(sentence)
                        sentence = []
                    continue
                sentence.append(line.split())
        if len(sentence) > 0:
            sentences.append(sentence)
        return sentences

    #the training set
    train = extract_sentences(r'D:\OneDrive\NER\Korpora\train.txt')

    #the test set
    test= extract_sentences(r'D:\OneDrive\NER\Korpora\test.txt')

    #Speichern der eingelesenen Daten als Liste aus Tupeln
    train_data = [[tuple(j) for j in i] for i in train]
    test_data = [[tuple(j) for j in i] for i in test]

    def get_embeddings(trainSent, testSent, devSent="", lang="en"):
        """
        Builds representations for each word based on GloVe embeddings
        :param trainSent:
        :param testSent:
        :param devSent:
        :param lang:
        :return:
        """
        labelSet = set()
        words = {}

        # unique words and labels in data
        for dataset in [trainSent, devSent, testSent]:
            for sentence in dataset:
                for token, pos, chunk, label in sentence:
                    # token ... token, char ... list of chars, label ... BIO labels
                    labelSet.add(label)
                    words[token.lower()] = True

        # mapping for labels
        label2Idx = {}
        for label in labelSet:
            label2Idx[label] = len(label2Idx)

        # read GLoVE word embeddings
        if lang == "en":
            path = "D:\GloVe word-embedding\glove.6B.50d.txt"
        elif lang == "es":
            path = ""
        else:
            raise Exception("Sorry, seems the requested language is not supported.")

        word2Idx = {}
        wordEmbeddings = []

        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                cols = line.strip().split(" ")
                word = cols[0]  # embedding word entry

                # Adds Zero-Vector as Padding and Random-Vector as Unknown-Token
                if len(word2Idx) == 0:  # add padding+unknown
                    word2Idx["PADDING_TOKEN"] = len(word2Idx)  #
                    vector = np.zeros(len(cols) - 1)  # zero vector for 'PADDING' word
                    wordEmbeddings.append(vector)

                    word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
                    #vector = np.random.uniform(-0.25, 0.25, len(cols) - 1)
                    # Average of all Wordembeddings to represent unknown tokens
                    vector = [-0.12920076, -0.28866628, -0.01224866, -0.05676644, -0.20210965, -0.08389011, 0.33359843,
                              0.16045167, 0.03867431, 0.17833012, 0.04696583, -0.00285802, 0.29099807, 0.04613704,
                              -0.20923874, -0.06613114, -0.06822549, 0.07665912, 0.3134014, 0.17848536, -0.1225775,
                              -0.09916984, -0.07495987, 0.06413227, 0.14441176, 0.60894334, 0.17463093, 0.05335403,
                              -0.01273871, 0.03474107, -0.8123879, -0.04688699, 0.20193407, 0.2031118, -0.03935686,
                              0.06967544, -0.01553638, -0.03405238, -0.06528071, 0.12250231, 0.13991883, -0.17446303,
                              -0.08011883, 0.0849521, -0.01041659, -0.13705009, 0.20127155, 0.10069408, 0.00653003,
                              0.01685157]
                    wordEmbeddings.append(vector)

                # Searches for
                if word.lower() in words:
                    vector = np.array([float(num) for num in cols[1:]])
                    wordEmbeddings.append(vector)  # word embedding vector
                    word2Idx[cols[0]] = len(word2Idx)  # corresponding word dict

        wordEmbeddings = np.array(wordEmbeddings)
        return word2Idx, wordEmbeddings

    word2idx, wordEmbeddings = get_embeddings(train_data, test_data)

    def get_pos(pos_tag):
        """
        Word-wise extraction of pos-tag
        :param pos_tag:
        :return:
        """
        if pos_tag == "NN":
            pos = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "JJ":
            pos = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "FW":
            pos = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "NNPS":
            pos = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "NNP":
            pos = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "NNS":
            pos = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "JJR":
            pos = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "TO":
            pos = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "VB":
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "IN":
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "VBD":
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "DT":
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "MD":
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "VBG":
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "VBN":
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "CD":
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "RB":
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "PRP":
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "VBZ":
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif pos_tag == "WRB":
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif pos_tag == "WDT":
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif pos_tag == "RP":
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif pos_tag == "WP":
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif pos_tag == "CC":
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif pos_tag == "RBR":
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        else:
            pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        return pos

    def get_casing(word):
        """
        Word-wise extraction of casing
        :param word:
        :return:
        """
        capital = "[(A-Z)+]"
        digit = "\d+"
        prefixes = "\b(Mr|St|Mrs|Ms|Dr)[.]*\b"
        suffixes = "\b(Inc|Ltd|Jr|Sr|Co)[.]*\b"
        others = "(-|&|%)"
        casing = [
             1 if re.search(capital,word)is not None and word.istitle() is False else 0,
             1 if re.search(digit,word)is not None else 0,
             1 if re.search(others,word)is not None else 0,
             1 if word.istitle() is True else 0,
             1 if re.search(prefixes, word)is not None else 0,
             1 if re.search(suffixes, word)is not None else 0
            ]
        return casing


    import re
    def word2features(sentence, i):
        """
        Word-wise-extraction of features
        :param sentence:
        :param i:
        :return:
        """
        features = []
        case_padding = [0]*6
        pos_padding = [0]*25

        # features für Wort-2, wenn das aktuelle Wort nicht an Anfang steht.
        if i > 1:
            word = sentence[i-2][0]
            postag = sentence[i-2][1]
            features.extend(get_casing(word))
            features.extend(get_pos(postag))
            if word in word2idx:
                features.extend(wordEmbeddings[word2idx[word]])
            else:
                features.extend(wordEmbeddings[word2idx["UNKNOWN_TOKEN"]])
        else:
            features.extend(case_padding)
            features.extend(pos_padding)
            features.extend(wordEmbeddings[word2idx["PADDING_TOKEN"]])

        # features für Wort-1, wenn das aktuelle Wort nicht an Anfang steht.
        if i > 0:
            word = sentence[i-1][0]
            postag = sentence[i-1][1]
            features.extend(get_casing(word))
            features.extend(get_pos(postag))
            if word in word2idx:
                features.extend(wordEmbeddings[word2idx[word]])
            else:
                features.extend(wordEmbeddings[word2idx["UNKNOWN_TOKEN"]])
        else:
            features.extend(case_padding)
            features.extend(pos_padding)
            features.extend(wordEmbeddings[word2idx["PADDING_TOKEN"]])


        #das aktuelle Wort
        word = sentence[i][0]
        postag = sentence[i][1]
        features.extend(get_casing(word))
        features.extend(get_pos(postag))
        if word in word2idx:
            features.extend(wordEmbeddings[word2idx[word]])
        else:
            features.extend(wordEmbeddings[word2idx["UNKNOWN_TOKEN"]])

        #features für wort+1, wenn ds aktuelle Wort nicht am Ende steht.
        if i < len(sentence)-1:
            word = sentence[i+1][0]
            postag = sentence[i+1][1]
            features.extend(get_casing(word))
            features.extend(get_pos(postag))
            if word in word2idx:
                features.extend(wordEmbeddings[word2idx[word]])
            else:
                features.extend(wordEmbeddings[word2idx["UNKNOWN_TOKEN"]])
        else:
            features.extend(case_padding)
            features.extend(pos_padding)
            features.extend(wordEmbeddings[word2idx["PADDING_TOKEN"]])

        #features für wort+2, wenn ds aktuelle Wort nicht am Ende steht.
        if i < len(sentence)-2:
            word = sentence[i+2][0]
            postag = sentence[i+2][1]
            features.extend(get_casing(word))
            features.extend(get_pos(postag))
            if word in word2idx:
                features.extend(wordEmbeddings[word2idx[word]])
            else:
                features.extend(wordEmbeddings[word2idx["UNKNOWN_TOKEN"]])
        else:
            features.extend(case_padding)
            features.extend(pos_padding)
            features.extend(wordEmbeddings[word2idx["PADDING_TOKEN"]])

        return features

    # sentence to features
    def sent2features(sentence):
        """
        Sentence-wise extraction of features
        :param sentence:
        :return:
        """
        return [word2features(sentence ,i) for i in range(len(sentence))]

    #extract the ner-tags for each word in each sentence (die gewünschten Ergebinisse)
    #extract ner-tag information for each word in one sentence
    def word_ner (sentence,i):
        """
        Word-wise extraction of labels
        :param sentence:
        :param i:
        :return:
        """
        ner_tag = sentence[i][3]
        if ner_tag == "B-PER":
            ner=[1,0,0,0,0,0,0,0,0]
        if ner_tag == "I-PER":
            ner=[0,1,0,0,0,0,0,0,0]
        if ner_tag == "B-LOC":
            ner=[0,0,1,0,0,0,0,0,0]
        if ner_tag == "I-LOC":
            ner=[0,0,0,1,0,0,0,0,0]
        if ner_tag == "B-ORG":
            ner=[0,0,0,0,1,0,0,0,0]
        if ner_tag == "I-ORG":
            ner=[0,0,0,0,0,1,0,0,0]
        if ner_tag == "B-MISC":
            ner=[0,0,0,0,0,0,1,0,0]
        if ner_tag == "I-MISC":
            ner=[0,0,0,0,0,0,0,1,0]
        if ner_tag == "O":
            ner=[0,0,0,0,0,0,0,0,1]
        return ner

    #sentence to ner_tag
    def sentence_ner_label (sentence):
        """
        Sentence-wise extraction of labels
        :param sentence:
        :return:
        """
        return [word_ner(sentence ,i) for i in range(len(sentence))]

    #define the training set
    x_train = [sent2features(sentence) for sentence in train_data]
    y_train = [sentence_ner_label(sentence) for sentence in train_data]
    #define the test set
    x_test = [sent2features(sentence) for sentence in test_data]
    y_test = [sentence_ner_label (sentence) for sentence in test_data]

    #transform data to matrix form
    x_train = [j for i in x_train for j in i]
    x_test = [j for i in x_test for j in i]
    y_train = [j for i in y_train for j in i]
    y_test = [j for i in y_test for j in i]

    # Daten in Numpy-Arrays umwandeln
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test

"""
TODO
def get_labels():
    pass


def array2index():
    pass
"""



