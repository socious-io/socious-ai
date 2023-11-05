from flask import Blueprint, request, jsonify
from .models import jobs_recommender

bp = Blueprint('recommender', __name__, url_prefix='/recommender')

TMP = 'impacts.html'


@bp.route('/jobs', methods=['POST'])
def recommend_jobs():
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400
    data = request.get_json()
    query = data.get('query', '')
    return jsonify({
        'jobs': jobs_recommender.predict(query)
    })
