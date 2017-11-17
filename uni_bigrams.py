# -*- encoding: utf8 -*-
import re

from pyvi.pyvi import ViTokenizer
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score
import datetime
import pandas as pd
import time
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def time_diff_str(t1, t2):
    """
    Calculates time durations.
    """
    diff = t2 - t1
    mins = int(diff / 60)
    secs = round(diff % 60, 2)
    return str(mins) + " mins and " + str(secs) + " seconds"

def load_model(model):
    print('loading model ...')
    if os.path.isfile(model):
        return joblib.load(model)
    else:
        return None

def clean_str_vn(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[~`@#$%^&*-+]", " ", string)
    def sharp(str):
        b = re.sub('\s[A-Za-z]\s\.', ' .', ' '+str)
        while (b.find('. . ')>=0): b = re.sub(r'\.\s\.\s', '. ', b)
        b = re.sub(r'\s\.\s', ' # ', b)
        return b
    string = sharp(string)
    string = re.sub(r" : ", ":", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def review_to_words(review, filename):
    """
    Function to convert a raw review to a string of words
    :param review
    :return: meaningful_words
    """
    # 1. Convert to lower case, split into individual words
    words = review.lower().split()
    # 2. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    with open(filename, "r") as f3:
        dict_data = f3.read()
        array = dict_data.splitlines()
    # 3. Remove stop words
    meaningful_words = [w for w in words if not w in array]

    # 4. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(meaningful_words)

def list_words(mes):
    words = mes.lower().split()
    return " ".join(words)

def review_to_words2(review, filename,n):
    with open(filename, "r") as f3:
        dict_data = f3.read()
        array = dict_data.splitlines()
    words = [' '.join(x) for x in ngrams(review, n)]
    meaningful_words = [w for w in words if not w in array]
    return build_sentence(meaningful_words)

def word_clean(array, review):
    words = review.lower().split()
    meaningful_words = [w for w in words if w in array]
    return " ".join(meaningful_words)

def print_words_frequency(train_data_features):
    # Take a look at the words in the vocabulary
    vectorizer = load_model('model/vectorizer.pkl')
    if vectorizer == None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)
    vocab = vectorizer.get_feature_names()
    print "Words in vocabulary:", vocab

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    print "Words frequency..."
    for tag, count in zip(vocab, dist):
        print count, tag

def ngrams(input, n):
  input = input.split(' ')
  output = []
  for i in range(len(input)-n+1):
    output.append(input[i:i+n])
  return output # output dang ['a b','b c','c d']

def ngrams2(input, n):
  input = input.split(' ')
  output = {}
  for i in range(len(input)-n+1):
    g = ' '.join(input[i:i+n])
    output.setdefault(g, 0)
    output[g] += 1
  return output # output la tu dien cac n-gram va tan suat cua no {'a b': 1, 'b a': 1, 'a a': 3}

def ngrams_array(arr,n):
    output = {}
    for x in arr:
        d = ngrams2(x, n)  # moi d la 1 tu dien
        for x in d:
            count = d.get(x)
            output.setdefault(x, 0)
            output[x] += count
    return output

def buid_dict(filename,arr,n,m):
    with open(filename, 'r') as f:
        ngram = ngrams_array(arr, n)
        for x in ngram:
            p = ngram.get(x)
            if p < m:
                f.write(x)

def build_sentence(input_arr):
    d = {}
    for x in range(len(input_arr)):
        d.setdefault(input_arr[x], x)
    chuoi = []
    for i in input_arr:
        x = d.get(i)
        if x == 0:
            chuoi.append(i)
        for j in input_arr:
            y = d.get(j)
            if y == x + 1:
                z = j.split(' ')
                chuoi.append(z[1])
    return " ".join(chuoi)

def load_data(filename):
    print "load data"
    col1 = []; col2 = []
    err = 0
    dict = 'dict_data/dict1'
    with open(filename, 'r') as f:
        for line in f:
            if line != "\n":
                # print line
                try:
                    label1, r, question = line.split(" ",2)
                    # print r

                    question = ViTokenizer.tokenize(unicode(question,encoding="utf-8"))
                    # print question
                    # question = clean_str_vn(question)
                    # question = review_to_words(question,'datavn/vietnamese-stopwords-dash.txt')
                    col1.append(label1)
                    col2.append(question)
                except:
                    err += 1

        col3 = []
        for q in col2:
            q = review_to_words2(q,dict,2)  # q la 1 cau
            q1 = [' '.join(x) for x in ngrams(q, 1)]  # q1:mang cac 1-grams
            q2 = [' '.join(x) for x in ngrams(q, 2)]  # q2: mang cac phan tu 2-grams
            q3 = [' '.join(x.replace(' ', '_') for x in q2)]
            y = q1 + q3
            z = " ".join(y)
            col3.append(z)
        # col1_train = col1[0:13000]
        # col3_train = col3[0:13000]
        # col1_test = col1[13000:len(col1)-1]
        # col3_test = col3[13000:len(col3)-1]
        # g = {"X_train":col3_train, "y_train": col1_train, "X_test": col3_test, "y_test": col1_test}
        print "load data done."
        print "err = ", err
    # return g
        d = {"label":col1, "question": col3}
        train = pd.DataFrame(d)
    return train

def load_data_train(filename):
    print "loading data"
    col1 = []; col2 = []
    err = 0
    with open(filename, 'r') as f:
        for line in f:
            if line != "\n":
                # print line
                try:
                    label1, r, question = line.split(" ",2)
                    question = ViTokenizer.tokenize(unicode(question,encoding="utf-8"))
                    col1.append(label1)
                    col2.append(question)
                except:
                    err += 1


        col3 = []
        for q in col2:
            q = review_to_words2(q,'dict_data/dict1',2)  # q la 1 cau
            q1 = [' '.join(x) for x in ngrams(q, 1)]  # q1:mang cac 1-grams
            q2 = [' '.join(x) for x in ngrams(q, 2)]  # q2: mang cac phan tu 2-grams
            q3 = [' '.join(x.replace(' ', '_') for x in q2)]
            y = q1 + q3
            z = " ".join(y)
            col3.append(z)
        g = {"label":col3, "question": col1}
        print "load data done."
        print "err = ", err
    return g
        # d = {"label1":col1, "question": col3}
        # train = pd.DataFrame(d)
    # return train

def dict_to_train_data():
    g1 = load_data_train('general_data/cleaned_report.txt')
    g2 = load_data_train('general_data/cleaned_report2.txt')
    labels = g1.get("label") + g2.get("label")
    questions = g1.get("question") + g2.get("question")
    d = {"label":labels, "question": questions}
    train = pd.DataFrame(d)
    return train

def training():
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)
    train = dict_to_train_data()
    train_text = train["question"].values
    vectorizer.fit(train_text)
    X_train = vectorizer.transform(train_text)
    X_train = X_train.toarray()
    y_train = train["label"]
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    fit1(X_train, y_train)

def fit1(X_train,y_train):
    print "fitting"
    uni_big = SVC(kernel='rbf', C=1000)
    uni_big.fit(X_train, y_train)
    joblib.dump(uni_big, 'model/uni_big.pkl')
    print "Done"

def predict_ex(mes):
    uni_big = load_model('model/uni_big.pkl')
    if uni_big == None:
        training()
    vectorizer = load_model('model/vectorizer.pkl')
    uni_big = load_model('model/uni_big.pkl')
    print "---------------------------"
    print "Training"
    print "---------------------------"

    # test_message = list_words(test_message) # lam thanh chu thuong
    clean_test_reviews = []
    clean_test_reviews.append(mes)
    d2 = {"message": clean_test_reviews}
    test = pd.DataFrame(d2)
    test_text = test["message"].values.astype('str')
    test_data_features = vectorizer.transform(test_text)
    test_data_features = test_data_features.toarray()
    # print test_data_features
    s = uni_big.predict(test_data_features)[0]
    return s


def train_main():
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=1000)
    train = load_data('general_data/cleaned_report.txt')
    test = load_data('general_data/cleaned_report2.txt')
    # print test
    # g = load_data('general_data/cleaned_report.txt', 'dict_data/dict1')
    # d1 = {"label": g.get("y_train"), "question": g.get("X_train")}
    # d2 = {"label": g.get("y_test"), "question": g.get("X_test")}
    # train = pd.DataFrame(d1)
    # test = pd.DataFrame(d2)
    print "Data dimensions:", train.shape
    print "List features:", train.columns.values
    print "First review:", train["label"][0], "|", train["question"][0]

    print "Data dimensions:", test.shape
    print "List features:", test.columns.values
    print "First review:", test["label"][0], "|", test["question"][0]
    train, test = train_test_split(train, test_size=0.2)

    train_text = train["question"].values
    test_text = test["question"].values

    vectorizer.fit(train_text)
    X_train = vectorizer.transform(train_text)
    X_train = X_train.toarray()
    y_train = train["label"]
    # print X_train

    X_test = vectorizer.transform(test_text)
    X_test = X_test.toarray()
    y_test = test["label"]


    print "---------------------------"
    print "Training"
    print "---------------------------"
    names = ["RBF SVC"]
    t0 = time.time()
    # iterate over classifiers

    clf = SVC(kernel='rbf', C=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print y_pred

    print " accuracy: %0.3f" % accuracy_score(y_test, y_pred)
    print " %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t0, time.time()))
    print "confuse matrix: \n", confusion_matrix(y_test, y_pred,
                                                 labels=["OT2", "OT1", "OT0", "CLOT2", "CLOT1", "CLNDT0", "CLNDT1", "CLNDT2", "DVOT2", "DVOT1", "DVOT0", "DVGV2", "DVCTSV2", "DVKT0", "DVKT2", "DVCSSV0", "DVCSSV2"])


if __name__ == '__main__':
    # train_main()
    training()
    # mes = raw_input("Nhap 1 cau: ")
    # kq = predict_ex(mes)
    # print kq
