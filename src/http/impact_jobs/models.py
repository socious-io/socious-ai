from src.models.impact_detector import ImpactDetectorModel
from src.db import DB


def jobs():
    return DB.fetch_lazy('''
  SELECT p.title, p.description, org.name as org_name, org.description as org_description 
  FROM projects p join organizations org on org.id=p.identity_id
  WHERE org.name IS NOT NULL AND org.name <> ''  AND org.verified_impact = true ORDER BY p.created_at
''', limit=10000)


impact_detector = ImpactDetectorModel(jobs)
