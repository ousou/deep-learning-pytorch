import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def read_pizza_data(train_data_perc):
    with open('../data/pizza/train.json') as data_file:
        raw_data = json.load(data_file)
        data_file.close()

    X = []
    Y = []
    for item in raw_data:
        X.append(item['request_title'] + ' ' + item['request_text'])
        Y.append(int(item['requester_received_pizza']))
    N = len(X)

    N_train = int(train_data_perc * N)

    X_train = X[:N_train]
    Y_train = Y[:N_train]

    X_test = X[N_train:]
    Y_test = Y[N_train:]

    X_train_vec, X_test_vec = transform_data(X_train, X_test)
    return np.array(X_train_vec.todense()), np.array(Y_train), np.array(X_test_vec.todense()), np.array(Y_test)


def transform_data(X_train, X_test):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(X_train), vectorizer.transform(X_test)

def main():
    X_train, Y_train, X_test, Y_test = read_pizza_data(0.8)

if __name__ == "__main__":
    main()
