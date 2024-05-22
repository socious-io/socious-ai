from src.models.impact_detector import ImpactDetectorModel
from src.db import DB


def jobs():
    return DB.fetch_lazy('''
  SELECT p.title, p.description, org.name as org_name, org.description as org_description 
  FROM projects p join organizations org on org.id=p.identity_id
  WHERE org.name IS NOT NULL AND org.name <> ''  AND org.verified_impact = true ORDER BY p.created_at
''', limit=10000)


def orgs():
    return DB.fetch_lazy('''
  SELECT id, bio, description, culture, social_causes, country FROM organizations WHERE 
                         verified_impact=true AND (description IS NOT NULL OR bio IS NOT NULL) ORDER BY created_at DESC
''', limit=2000)


def impact_detector(name):
    class C(ImpactDetectorModel):

        @property
        def name(self):
            return name
    return C


impact_job_detector = impact_detector('impact_job_detector')(jobs)
impact_org_detector = impact_detector('impact_org_detector')(orgs)
