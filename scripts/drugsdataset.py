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
from nltk.stem import SnowballStemmer


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




    def preprocess_opinion(self,
                           remove_stopwords=False,
                           stemming=False):
        def _preprocess_opinion(obj, func):
            obj.train['opinion'] = obj.train['opinion'].apply(lambda x: func(x))
            obj.test['opinion'] = obj.test['opinion'].apply(lambda x: func(x))
            return

        # TODO: text cleaning( ie.: I'am -> I am), punctuation?
        # Opinion to lowercase
        _preprocess_opinion(self, lambda x: x.lower())
        print("Lowercase done...")

        # Optional removing stopwords
        stop_words = ['the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 'which', 'this', 'that',
                      'these', 'those', 'then',
                      'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
                      'to']
        if remove_stopwords:
            def _remove_stopwords(text):
                text = text.split()
                text = [w for w in text if not w in stop_words]
                text = " ".join(text)
                return text
            _preprocess_opinion(self, lambda x: _remove_stopwords(x))
            print("Stopwords removed...")
        # English stemming
        if stemming:
            def _stem_words(text):
                text = text.split()
                stemmer = SnowballStemmer('english')
                stemmed_words = [stemmer.stem(word) for word in text]
                text = " ".join(stemmed_words)
                return text
            _preprocess_opinion(self, lambda x: _stem_words(x))
            print("Opinions stemmed...")

        # Preprocessing done
        print("Preprocessing done!")
        return

    def feature_engineering(self):
        """
        Classic features for NLP tasks like Tfidf or word2vec
        :return:
        """

    def create_bag_of_words(self, max_features=None):
        count_vectorizer = CountVectorizer(analyzer="word",
                                           tokenizer=nltk.word_tokenize,
                                           preprocessor=None,
                                           stop_words='english',
                                           max_features=max_features)
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