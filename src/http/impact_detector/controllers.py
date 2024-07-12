import requests
from flask import Blueprint, request, jsonify, render_template
from .models import impact_job_detector, impact_org_detector

bp = Blueprint('impacts', __name__, url_prefix='/impacts')

TMP = 'impacts.html'


@bp.route('/jobs/accuracy', methods=['GET'])
def accuracy():
    return jsonify({
        'accuracy': impact_job_detector.accuracy
    })


@bp.route('/jobs', methods=['POST'])
def verify_jobs():
    if impact_job_detector.status != impact_job_detector.STATUS_TRAINED:
        return jsonify({"error": "impact detector is not ready to use"}), 400
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    query = data.get('query', '')
    return jsonify({
        'predicts': [1 if item else 0 for item in impact_job_detector.predict(query)]
    })


@bp.route('/orgs', methods=['POST'])
def verify_orgs():
    if impact_org_detector.status != impact_org_detector.STATUS_TRAINED:
        return jsonify({"error": "impact detector is not ready to use"}), 400
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    query = data.get('query', '')
    return jsonify({
        'predicts': [1 if item else 0 for item in impact_org_detector.predict(query)]
    })


@bp.route('/jobs/one', methods=['POST'])
def verify_jobs_one():
    if impact_job_detector.status != impact_job_detector.STATUS_TRAINED:
        return jsonify({"error": "impact detector is not ready to use"}), 400
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    query = data.get('description', '')
    return jsonify({
        'result': True if impact_job_detector.predict(query)[0] else False
    })


@bp.route('/orgs/one', methods=['POST'])
def verify_orgs_one():
    if impact_org_detector.status != impact_org_detector.STATUS_TRAINED:
        return jsonify({"error": "impact detector is not ready to use"}), 400
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    query = data.get('description', '')
    return jsonify({
        'result': True if impact_org_detector.predict(query)[0] else False
    })


@bp.route('/verify.html', methods=['POST'])
def verify_html():
    try:
        link = request.form.get('job_link')
        req = requests.get(link)
        return render_template(TMP, **{'impact': impact_job_detector.is_impact_job(req.text), 'form': {}, 'accuracy': impact_detector.accuracy})
    except Exception as err:
        return render_template(TMP, **{'error': err, 'form': {}, 'accuracy': impact_job_detector.accuracy})
