from random import sample
import joblib
import yake
import pandas as pd
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
    YAKE = yake.KeywordExtractor(n=3, dedupLim=0.9, top=50, features=None)

    K_N_COUNT = 8

    STATUS_INIT = 'init'
    STATUS_TRAINING = 'training'
    STATUS_TRAINED = 'trained'

    TEST_DATA_SELECT_PERCENT = 30
    STOP_WORDS = set(stopwords.words('english'))
    LEMMATIZER = WordNetLemmatizer()

    VECTORIZER = TfidfVectorizer()

    @property
    def name(self):
        return str(self.__class__)

    @property
    def model_name(self):
        return '%s_model.pkl' % self.name

    @property
    def vectorizer_name(self):
        return '%s_vectorizer.pkl' % self.name

    def __init__(self, data_loader_func) -> None:
        self.data_loader_func = data_loader_func
        self.accuracy = 0
        self.model = None
        self.status = self.STATUS_INIT

    def load_data(self):
        data = self.data_loader_func()
        print('Fetched %d of data for %s' % (len(data), self.name))
        length = len(data)
        if length < 10:
            raise ValueError('data length is too low')
        test_sample_count = int(length * self.TEST_DATA_SELECT_PERCENT / 100)
        train_sample_count = length - test_sample_count
        self.data = pd.DataFrame(sample(data, train_sample_count))
        self.test_data = pd.DataFrame(sample(data, test_sample_count))

    def clean_text(self, text):
        text = re.sub('<.*?>', '', text)  # Remove HTML tags
        text = re.sub('[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        text = re.sub(r"_+", " ", text)
        return text

    def preprocess_text(self, text):
        text = self.clean_text(text)
        keywords = self.YAKE.extract_keywords(text)
        result = ' '.join([k[0] for k in keywords])
        words = set(result.split())
        return ' '.join(words)

    def obj_to_text(self, obj):
        values = [
            ' '.join(val) if isinstance(val, list) else val
            for key, val in obj.items()
            if key != 'id' and (isinstance(val, str) or isinstance(val, list))
        ]
        return ' '.join(values)

    def get_train_model(self):
        raise NotImplemented

    def train(self, force=False):
        if self.status == self.STATUS_TRAINING:
            return

        self.status = self.STATUS_TRAINING
        self.load_data()
        if not force:
            try:
                model = joblib.load(self.model_name)
                self.VECTORIZER = joblib.load(self.vectorizer_name)
                self.model = model
                self.status = self.STATUS_TRAINED
                return
            except Exception:
                pass

        proccessed_data = [self.preprocess_text(
            self.obj_to_text(item)) for _, item in self.data.iterrows()]

        tfidf_matrix = self.VECTORIZER.fit_transform(proccessed_data)
        self.model = self.get_train_model()
        self.model.fit(tfidf_matrix)
        joblib.dump(self.model, self.model_name)
        joblib.dump(self.VECTORIZER, self.vectorizer_name)
        self.status = self.STATUS_TRAINED

    def predict(self, query):
        if not isinstance(query, (list, tuple, np.ndarray)):
            query = [query]

        query_data = pd.DataFrame(query)

        proccessed_query_data = [self.preprocess_text(
            self.obj_to_text(item)) for _, item in query_data.iterrows()]

        query_matrix = self.VECTORIZER.transform(proccessed_query_data)
        _, indices = self.model.kneighbors(query_matrix)
        elements = list(dict.fromkeys(
            element for sublist in indices for element in sublist))
        return list(self.data.iloc[elements]['id'].values)
