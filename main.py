from src.db import DB
from src.detector import ImpactJobsDetector

jobs = DB.fetch_lazy('''
  SELECT p.title, p.description, p.skills, org.name as org_name, org.description as org_description 
  FROM projects p join organizations org on org.id=p.identity_id
''')


detector = ImpactJobsDetector(jobs)

print(detector.is_impact_job(jobs[1]))
