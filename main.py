from src.models.job_recommender import TrainModel
from src.db import DB

if __name__ == "__main__":
    jobs = DB.fetch_lazy('''
    SELECT p.id, p.title, p.description, org.name as org_name, org.description as org_description 
    FROM projects p join organizations org on org.id=p.identity_id
    WHERE org.name IS NOT NULL OR org.name <> '' ORDER BY RANDOM()
    ''', limit=10000)

    query_jobs = DB.fetch_lazy('''
    SELECT p.title, p.description, org.name as org_name, org.description as org_description 
    FROM projects p join organizations org on org.id=p.identity_id
    WHERE org.name IS NOT NULL OR org.name <> '' ORDER BY RANDOM()
    ''', limit=3)
    model = TrainModel(jobs)
    model.train()
    print(model.predict(query_jobs))
