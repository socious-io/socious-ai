from flask import Blueprint, request, jsonify
from src.config import config
import threading
from .models import jobs_recommender, talents_recommender, orgs_recommender

bp = Blueprint('recommender', __name__, url_prefix='/recommender')

TMP = 'impacts.html'


@bp.route('/retrain', methods=['GET'])
def retrain():
    token = request.args.get('token')
    if token != config.admin_token:
        return jsonify({"error": "not valid token"}), 401

    model = request.args.get('model')
    if (model == 'jobs'):
        threading.Thread(target=jobs_recommender.train, args=(True, )).start()
    if (model == 'talents'):
        threading.Thread(target=talents_recommender.train,
                         args=(True, )).start()
    if (model == 'orgs'):
        threading.Thread(target=orgs_recommender.train, args=(True, )).start()
    return jsonify({
        'msg': 'success'
    })


@bp.route('/jobs', methods=['POST'])
def recommend_jobs():
    if jobs_recommender.status != jobs_recommender.STATUS_TRAINED:
        return jsonify({"error": "recommender is not ready to use"}), 400
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    query = data.get('query', '')
    predicts = jobs_recommender.predict(query)
    interests = data.get('intrests', None)
    if interests:
        interests_predicts = jobs_recommender.predict_by_ids(interests)
        predicts = list(set(predicts + interests_predicts))
    excludes = data.get('excludes', None)
    if excludes:
        excludes_predicts = jobs_recommender.predict_by_ids(excludes)
        excludes_predicts += excludes
        predicts = [item for item in predicts if item not in excludes_predicts]

    return jsonify({
        'jobs': predicts
    })


@bp.route('/talents', methods=['POST'])
def recommend_talents():
    if talents_recommender.status != talents_recommender.STATUS_TRAINED:
        return jsonify({"error": "recommender is not ready to use"}), 400
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    query = data.get('query', '')
    return jsonify({
        'talents': talents_recommender.predict(query)
    })


@bp.route('/orgs', methods=['POST'])
def recommend_orgs():
    if orgs_recommender.status != orgs_recommender.STATUS_TRAINED:
        return jsonify({"error": "recommender is not ready to use"}), 400
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    query = data.get('query', '')
    return jsonify({
        'orgs': orgs_recommender.predict(query)
    })
