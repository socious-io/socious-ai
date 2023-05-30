from itertools import compress
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
from langdetect import detect
from schema import Schema, And, Use

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')


class ImpactDetector:
    STOP_WORDS = set(stopwords.words('english'))
    CLEANER = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    VECTORIZER = TfidfVectorizer()
    MODEL_NAME = 'impact_jobs_detector.pkl'

    JOB_SCHEMA = Schema({
        'title': And(str, len),
        'description':  And(str, len),
        'org_name': Use(str),
        'org_description':  Use(str),
        # 'skills': Or(None, [And(str, len)])
    })

    def __init__(self, jobs) -> None:
        self.validate_jobs(jobs)
        self.jobs = jobs
        self.model = None
        self.train()

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

    def preprocess_text(self, text):
        text = re.sub(self.CLEANER, '', text)
        word_tokens = word_tokenize(text)
        filtered_text = [
            word for word in word_tokens if word.casefold() not in self.STOP_WORDS]
        return " ".join(filtered_text)

    def is_english(self, text):
        try:
            language = detect(text)
        except Exception:
            return False
        if language == 'en':
            return True
        return False

    def train(self, force=False):
        if not force:
            try:
                model = joblib.load(self.MODEL_NAME)
                self.model = model
                return
            except Exception:
                pass

        print('Start training with %d of jobs ....' % len(self.jobs))

        corpus = [self.convert_job_to_text(job) for job in self.jobs]
        corpus = [self.preprocess_text(text) for text in corpus]
        corpus = list(filter(self.is_english, corpus))
        # corpus = list(compress(self.is_english, corpus))
        print('%d of jobs detected as EN to train' % len(corpus))
        # Vectorize the text
        self.VECTORIZER.fit(corpus)
        dataset = self.VECTORIZER.fit_transform(corpus)

        # Create and train a one-class SVM
        self.model = OneClassSVM(gamma='auto').fit(dataset)
        joblib.dump(self.model, self.MODEL_NAME)

    def is_impact_job(self, job):
        self.validate_jobs([job])
        text_job = self.convert_job_to_text(job)
        text_job = self.preprocess_text(text_job)
        new = self.VECTORIZER.transform([text_job])
        prediction = self.model.predict(new)[0]
        return prediction == 1
