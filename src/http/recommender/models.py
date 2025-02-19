from src.models.job_recommender import JobRecommender
from src.models.talent_recommender import TalentRecommender
from src.models.org_recommender import OrgRecommender
from src.db import DB


def jobs():
    return DB.fetch_lazy('''
  SELECT p.id, p.title, p.description, p.country, org.name as org_name, org.description as org_description, p.causes_tags 
  FROM projects p join organizations org on org.id=p.identity_id
  WHERE (org.name IS NOT NULL OR org.name <> '') AND (expires_at IS NULL OR expires_at > NOW()) ORDER BY p.created_at DESC
''')


def users():
    return DB.fetch_lazy('''
  SELECT id, bio, mission, skills, social_causes, country FROM users ORDER BY created_at DESC
''')


def orgs():
    return DB.fetch_lazy('''
  SELECT id, bio, description, culture, social_causes, country FROM organizations ORDER BY created_at DESC
''')


jobs_recommender = JobRecommender(jobs)
talents_recommender = TalentRecommender(users)
orgs_recommender = OrgRecommender(orgs)
