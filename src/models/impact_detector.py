import hashlib
from multiprocessing import cpu_count
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import joblib
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import re
from langdetect import detect
from schema import Schema, And, Use
from sklearn.metrics import classification_report
import string


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class ImpactDetector:
    STOP_WORDS = set(stopwords.words('english'))
    LEMMATIZER = WordNetLemmatizer()
    PROCCESSED_TEXTS_DB = 'processed_texts.db'
    CLEANER = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

    SUMMARIZER_MODEL_NAME = 't5-small'
    TOKENIZER = T5Tokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)
    SUMMARIZER = T5ForConditionalGeneration.from_pretrained(
        SUMMARIZER_MODEL_NAME)

    VECTORIZER = TfidfVectorizer()
    MODEL_NAME = 'impact_jobs_detector.pkl'
    VECTORIZER_NAME = 'tfidf_vectorizer.pkl'

    JOB_SCHEMA = Schema({
        'title': And(str, len),
        'description':  And(str, len),
        'org_name': Use(str),
        'org_description':  Use(str),
        # 'skills': Or(None, [And(str, len)])
    })

    def __init__(self, jobs, test_jobs) -> None:
        self.test_jobs = pd.DataFrame(test_jobs)
        self.accuracy = 0
        self.validate_jobs(jobs)
        self.jobs = pd.DataFrame(jobs)
        self.model = None

    def validate_jobs(self, jobs):
        validated = Schema([self.JOB_SCHEMA]).validate(jobs)
        if not validated:
            raise Exception('not valid job')

    def convert_job_to_text(self, job):
        return '%s %s %s %s %s' % (
            job.get('org_name', ''),
            job.get('org_description', ''),
            job['title'],
            job['description'],
            ' '.join(job.get('skills') or [])
        )

    def create_unique_id(self, text):
        return hashlib.sha256(text.encode()).hexdigest()

    def summaries(self, text):
        if len(text) < 100:
            return text

        inputs = self.TOKENIZER.encode(
            text, return_tensors="pt", max_length=512)
        outputs = self.SUMMARIZER.generate(
            inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        return self.TOKENIZER.decode(outputs[0])

    def clean_tags(self, text):
        soup = BeautifulSoup(text, "html.parser")
        cleaned_html = soup.get_text()
        TAG_RE = re.compile(r'<[^>]+>')
        return TAG_RE.sub('', cleaned_html)

    def preprocess_text(self, text):
        text = self.clean_tags(text)
        text = re.sub(self.CLEANER, '', text)
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
        processed = None
        try:
            processed = self.summaries(text)
        except Exception as e:
            print('Summerize error %s ' % e)
            processed = text

        return self.clean_tags(processed)

    def is_english(self, text):
        try:
            language = detect(text)
        except Exception:
            return False
        if language == 'en':
            return True
        return False

    def apply_parallel(self, df, func):
        return joblib.Parallel(n_jobs=cpu_count())(joblib.delayed(func)(i) for i in df)

    def train(self, force=False):
        if not force:
            try:
                model = joblib.load(self.MODEL_NAME)
                self.VECTORIZER = joblib.load(self.VECTORIZER_NAME)
                self.model = model
                self.evaluate()
                return
            except Exception:
                pass

        print('Start training with %d of jobs ....' % len(self.jobs))
        self.jobs['description'] = self.apply_parallel(
            self.jobs['description'], self.preprocess_text)

        self.jobs['org_description'] = self.apply_parallel(
            self.jobs['org_description'], self.preprocess_text)

        """         result = joblib.Parallel(n_jobs=10, verbose=10)(
            joblib.delayed(self.preprocess)(job) for job in tqdm(self.jobs))

        corpus = list(filter(self.is_english, result))
        print('%d of jobs detected as EN to train' % len(corpus))
        # Vectorize the text
        self.VECTORIZER.fit(corpus)
        dataset = self.VECTORIZER.transform(corpus)
        param_grid = {'nu': [0.01, 0.1, 0.3, 0.5, 0.7, 0.9],
                      'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                      'gamma': [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                      'degree': [2, 3, 4, 5]
                      }

        # Do the grid search manually
        best_score = float('inf')
        best_params = None
        for params in ParameterGrid(param_grid):
            svm = OneClassSVM(**params)
            svm.fit(dataset)
            score = self.anomaly_score(dataset, svm)
            if score < best_score:
                best_score = score
                best_params = params

        self.model = OneClassSVM(**best_params)
        self.model.fit(dataset)

        joblib.dump(self.model, self.MODEL_NAME)
        joblib.dump(self.VECTORIZER, self.VECTORIZER_NAME)
        self.evaluate() """

    def anomaly_score(self, X, estimator):
        decision_function = estimator.decision_function(X)
        return np.mean(decision_function)

    def evaluate(self):
        result = joblib.Parallel(n_jobs=10, verbose=10)(
            joblib.delayed(self.preprocess_text(job) for job in tqdm(self.test_jobs)))
        corpus = list(filter(self.is_english, result))
        results = self.VECTORIZER.transform(corpus)
        predictions = self.model.predict(results)

        # Use classification report instead of accuracy
        report = classification_report([1]*len(corpus), predictions)
        print(report)

        self.accuracy = accuracy_score([1]*len(corpus), predictions)
        # Write the report to a file
        with open("classification_report.txt", "w") as f:
            f.write(report)
            f.write("\nAccuracy: " + str(self.accuracy))

    def is_impact_job(self, job):
        if type(job) == dict:
            self.validate_jobs([job])
            text_job = self.convert_job_to_text(job)
        else:
            text_job = job

        text_job = self.preprocess_text(job)
        print(text_job, '---------------------@@@-')
        new = self.VECTORIZER.transform([text_job])
        prediction = self.model.predict(new)[0]
        return prediction == 1
