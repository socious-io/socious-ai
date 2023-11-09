from .base_recommender import TrainModel
from sklearn.neighbors import NearestNeighbors


class TalentRecommender(TrainModel):

    @property
    def name(self):
        return 'talen_recommender'

    def get_train_model(self):
        return NearestNeighbors(n_neighbors=8)
