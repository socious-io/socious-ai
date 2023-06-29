import shelve
import hashlib
import joblib
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
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

# Download necessary NLTK data
if __name__ == "__main__":
    # Download nltk data when script is run as main module
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')  # for lemmatization


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
        self.test_jobs = test_jobs
        self.accuracy = 0
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
        inputs = self.TOKENIZER.encode(
            text, return_tensors="pt", max_length=512)
        outputs = self.SUMMARIZER.generate(
            inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        return self.TOKENIZER.decode(outputs[0])

    def preprocess_text(self, text):
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
                self.evaluate()
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
        dataset = self.VECTORIZER.transform(corpus)

        # Create and train a one-class SVM
        # GridSearchCV to tune hyperparameters
        param_grid = {'gamma': ['scale', 'auto'], 'nu': [0.5, 0.7, 0.9]}
        grid_search = GridSearchCV(OneClassSVM(), param_grid, cv=5)
        grid_search.fit(dataset)

        self.model = grid_search.best_estimator_
        joblib.dump(self.model, self.MODEL_NAME)
        joblib.dump(self.VECTORIZER, self.VECTORIZER_NAME)
        self.evaluate()

    def evaluate(self):
        corpus = [self.convert_job_to_text(job) for job in self.test_jobs]
        result = joblib.Parallel(n_jobs=10, verbose=10)(
            joblib.delayed(self.preprocess_text)(text) for text in tqdm(corpus))
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
        self.validate_jobs([job])
        text_job = self.convert_job_to_text(job)
        text_job = self.preprocess_text(text_job)
        new = self.VECTORIZER.transform([text_job])
        prediction = self.model.predict(new)[0]
        return prediction == 1
