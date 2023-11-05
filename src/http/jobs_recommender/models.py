from src.models.job_recommender import TrainModel
from src.db import DB


jobs = DB.fetch_lazy('''
  SELECT p.id, p.title, p.description, org.name as org_name, org.description as org_description 
  FROM projects p join organizations org on org.id=p.identity_id
  WHERE org.name IS NOT NULL OR org.name <> '' ORDER BY p.created_at
''')


jobs_recommender = TrainModel(jobs)
