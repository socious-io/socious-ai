from .controllers import bp
from .models import jobs_recommender, talents_recommender, orgs_recommender


mod = bp
ai_models = [
    jobs_recommender,
    talents_recommender,
    orgs_recommender
]
