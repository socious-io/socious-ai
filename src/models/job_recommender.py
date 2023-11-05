from random import sample
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
import nltk
import string
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class TrainModel:
    K_N_COUNT = 8
    TEST_DATA_SELECT_PERCENT = 30
    STOP_WORDS = set(stopwords.words('english'))
    LEMMATIZER = WordNetLemmatizer()
    PROCCESSED_TEXTS_DB = 'processed_texts.db'

    SUMMARIZER_MODEL_NAME = 't5-small'
    TOKENIZER = T5Tokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)
    SUMMARIZER = T5ForConditionalGeneration.from_pretrained(
        SUMMARIZER_MODEL_NAME)

    VECTORIZER = TfidfVectorizer()
    MODEL_NAME = 'job_recommender.pkl'
    VECTORIZER_NAME = 'job_recommender_vectorizer.pkl'

    def __init__(self, data) -> None:
        length = len(data)
        if length < 10:
            raise ValueError('data length is too low')
        test_sample_count = int(length * self.TEST_DATA_SELECT_PERCENT / 100)
        train_sample_count = length - test_sample_count

        self.test_data = pd.DataFrame(sample(data, test_sample_count))
        self.data = pd.DataFrame(sample(data, train_sample_count))
        self.accuracy = 0
        self.model = None

    def clean_text(self, text):
        text = re.sub('<.*?>', '', text)  # Remove HTML tags
        text = re.sub('[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        text = re.sub(r"_+", " ", text)
        return text

    def preprocess_text(self, text):

        text = self.clean_text(text)
        word_tokens = word_tokenize(text)
        # Lemmatization
        lemmatized_words = [self.LEMMATIZER.lemmatize(
            word) for word in word_tokens]
        # Remove punctuation
        words_without_punct = [
            word for word in lemmatized_words if word not in string.punctuation]
        filtered_text = [
            word for word in words_without_punct if word.casefold() not in self.STOP_WORDS]

        text = " ".join(filtered_text).lower()
        return self.clean_text(text)

    def extract_keywords(self, matrix):
        keywords = []
        for idx in range(matrix.shape[0]):
            vector = matrix.getrow(idx)
            sorted_indices = vector.toarray().ravel().argsort()[-50:][::-1]
            keywords.append(' '.join([self.VECTORIZER.get_feature_names_out()[i]
                                      for i in sorted_indices]))
        return keywords

    def obj_to_text(self, obj):
        values = [
            ' '.join(val) if isinstance(val, list) else val
            for key, val in obj.items()
            if key != 'id' and (isinstance(val, str) or isinstance(val, list))
        ]
        return ' '.join(values)

    def train(self, force=False):
        if not force:
            try:
                model = joblib.load(self.MODEL_NAME)
                self.VECTORIZER = joblib.load(self.VECTORIZER_NAME)
                self.model = model
                return
            except Exception:
                pass
        proccessed_data = [self.preprocess_text(
            self.obj_to_text(item)) for _, item in self.data.iterrows()]

        tfidf_matrix = self.VECTORIZER.fit_transform(proccessed_data)
        tfidf_matrix = self.VECTORIZER.transform(
            self.extract_keywords(tfidf_matrix))
        self.model = NearestNeighbors(n_neighbors=self.K_N_COUNT)
        self.model.fit(tfidf_matrix)
        joblib.dump(self.model, self.MODEL_NAME)
        joblib.dump(self.VECTORIZER, self.VECTORIZER_NAME)

    def predict(self, query):
        if not isinstance(query, (list, tuple, np.ndarray)):
            query = [query]

        query_data = pd.DataFrame(query)
        proccessed_query_data = [self.preprocess_text(
            self.obj_to_text(item)) for _, item in query_data.iterrows()]

        query_matrix = self.VECTORIZER.transform(proccessed_query_data)
        query_matrix = self.VECTORIZER.transform(
            self.extract_keywords(query_matrix))

        _, indices = self.model.kneighbors(query_matrix)
        elements = list(dict.fromkeys(
            element for sublist in indices for element in sublist))
        return list(self.data.iloc[elements]['id'].values)
