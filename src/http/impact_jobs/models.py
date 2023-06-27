from src.models.impact_detector import ImpactDetector
from src.db import DB


jobs = DB.fetch_lazy('''
  SELECT p.title, p.description, org.name as org_name, org.description as org_description 
  FROM projects p join organizations org on org.id=p.identity_id
  WHERE org.name IS NOT NULL OR org.name <> '' ORDER BY p.created_at
''', limit=30000)


test_jobs = DB.fetch_lazy('''
  SELECT p.title, p.description, org.name as org_name, org.description as org_description 
  FROM projects p join organizations org on org.id=p.identity_id
  WHERE org.name IS NOT NULL OR org.name <> '' ORDER BY RANDOM()
''', limit=1000)


impact_detector = ImpactDetector(jobs, test_jobs)
