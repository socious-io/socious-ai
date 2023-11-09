from .base_recommender import TrainModel
from sklearn.neighbors import NearestNeighbors


class OrgRecommender(TrainModel):

    @property
    def name(self):
        return 'orgs_recommender'

    def get_train_model(self):
        return NearestNeighbors(n_neighbors=8)
