import requests
from flask import Blueprint, request, jsonify, render_template
from .models import impact_detector

bp = Blueprint('impacts', __name__, url_prefix='/impacts')

TMP = 'impacts.html'


@bp.route('/jobs/accuracy', methods=['GET'])
def accuracy():
    return jsonify({
        'accuracy': impact_detector.accuracy
    })


@bp.route('/jobs', methods=['POST'])
def verify():
    if impact_detector.status != impact_detector.STATUS_TRAINED:
        return jsonify({"error": "impact detector is not ready to use"}), 400
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    query = data.get('query', '')

    return jsonify({
        'predict': [False if item == -1 else True for item in impact_detector.predict(query)]
    })


@bp.route('/verify.html', methods=['POST'])
def verify_html():
    try:
        link = request.form.get('job_link')
        req = requests.get(link)
        return render_template(TMP, **{'impact': impact_detector.is_impact_job(req.text), 'form': {}, 'accuracy': impact_detector.accuracy})
    except Exception as err:
        return render_template(TMP, **{'error': err, 'form': {}, 'accuracy': impact_detector.accuracy})
