from .base_recommender import TrainModel
from sklearn.neighbors import NearestNeighbors


class JobRecommender(TrainModel):

    @property
    def name(self):
        return 'jobs_recommender'

    def get_train_model(self):
        return NearestNeighbors(n_neighbors=8)
