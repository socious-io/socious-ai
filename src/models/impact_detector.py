import shelve
import hashlib
from tqdm import tqdm
from transformers import pipeline
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
if __name__ == "__main__":
    # Download nltk data when script is run as main module
    nltk.download('punkt')
    nltk.download('stopwords')
    summarizer = pipeline(
        "summarization", model="sshleifer/distilbart-cnn-12-6")


class ImpactDetector:
    STOP_WORDS = set(stopwords.words('english'))
    PROCCESSED_TEXTS_DB = 'processed_texts.db'
    CLEANER = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

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

    def __init__(self, jobs) -> None:
        """ self.manager = Manager()
        self.db_lock = self.manager.Lock() """
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

    def create_unique_id(self, text):
        return hashlib.sha256(text.encode()).hexdigest()

    def summaries(self, text):

        if len(text) <= 50:
            return text
        chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
        summaries = []
        for chunk in chunks:
            max_length = int(len(chunk.split()) * 0.3)
            min_length = int(len(chunk.split()) * 0.1)
            if max_length < 10:
                max_length = 10
            if min_length < 5:
                min_length = 5
            summaries.append(summarizer(
                chunk, max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]['summary_text'])

        return " ".join(summaries)

    def preprocess_text(self, text):
        text = re.sub(self.CLEANER, '', text)
        word_tokens = word_tokenize(text)
        filtered_text = [
            word for word in word_tokens if word.casefold() not in self.STOP_WORDS]

        text = " ".join(filtered_text).lower()
        # id = self.create_unique_id(text)
        processed = None
        try:
            processed = self.summaries(text)
        except Exception as e:
            print('Summerize error %s ' % e)
            processed = text

        return processed

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
                self.VECTORIZER = joblib.load(self.VECTORIZER_NAME)
                self.model = model
                return
            except Exception:
                pass

        print('Start training with %d of jobs ....' % len(self.jobs))

        corpus = [self.convert_job_to_text(job) for job in self.jobs]
        result = joblib.Parallel(n_jobs=10, verbose=10)(
            joblib.delayed(self.preprocess_text)(text) for text in tqdm(corpus))

        corpus = list(filter(self.is_english, result))
        print('%d of jobs detected as EN to train' % len(corpus))
        # Vectorize the text
        self.VECTORIZER.fit(corpus)
        dataset = self.VECTORIZER.fit_transform(corpus)

        # Create and train a one-class SVM
        self.model = OneClassSVM(gamma='auto').fit(dataset)
        joblib.dump(self.model, self.MODEL_NAME)
        joblib.dump(self.VECTORIZER, self.VECTORIZER_NAME)

    def is_impact_job(self, job):
        self.validate_jobs([job])
        text_job = self.convert_job_to_text(job)
        text_job = self.preprocess_text(text_job)
        new = self.VECTORIZER.transform([text_job])
        prediction = self.model.predict(new)[0]
        return prediction == 1
