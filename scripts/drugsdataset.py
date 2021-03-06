"""
Class DrugsDataset implements:
- data reading
- preprocessing
- feature engineering
- submission prediction
"""
import pandas as pd
import numpy as np
import os
import nltk
from sklearn.feature_extraction.text import CountVectorizer


class DrugsDataset:
    def __init__(self, data_path):
        """

        :param data_path:
        """
        self.train = pd.read_csv(os.path.join(data_path, "Train.csv"), sep=";")
        self.test = pd.read_csv(os.path.join(data_path, "TestX.csv"), sep=";")
        self.sample_submission = pd.read_csv(os.path.join(data_path, "JANLOS.txt"), sep=" ")

        return

    #TODO:
    def create_meta_features(self):
        """
        Features useful in NLP tasks like text length or wordcount
        :return:
        """
        pass

    def preprocess_opinion(self):
        pass

    def feature_engineering(self):
        """
        Classic features for NLP tasks like Tfidf or word2vec
        :return:
        """

    def create_bag_of_words(self):
        count_vectorizer = CountVectorizer(analyzer="word",
                                           tokenizer=nltk.word_tokenize,
                                           preprocessor=None,
                                           stop_words='english',
                                           max_features=None)
        bag_of_words_train = count_vectorizer.fit_transform(self.train['opinion'])
        bag_of_words_test = count_vectorizer.transform(self.test['opinion'])
        return bag_of_words_train, bag_of_words_test

def main():
    DATA_PATH = "/home/jakubkala/IAD/semestr-2/zmum/ZMUM_project/data"
    drugsdataset = DrugsDataset(DATA_PATH)
    print(drugsdataset.train.head())
    return

if __name__ == "__main__":
    main()