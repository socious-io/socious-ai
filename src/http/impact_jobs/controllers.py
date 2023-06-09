import requests
from flask import Blueprint, request, jsonify, render_template
from .models import impact_detector

bp = Blueprint('impact_jobs', __name__, url_prefix='/impacts')

TMP = 'impacts.html'


@bp.route('', methods=['GET'])
def home():
    return render_template(TMP, **{'form': {}, 'accuracy': impact_detector.accuracy})


@bp.route('/verify.json', methods=['POST'])
def verify():
    try:
        return jsonify({'impact': impact_detector.is_impact_job(request.json)})
    except Exception as err:
        return jsonify({'error': str(err)}), 400


@bp.route('/verify.html', methods=['POST'])
def verify_html():
    try:
        link = request.form.get('job_link')
        req = requests.get(link)
        return render_template(TMP, **{'impact': impact_detector.is_impact_job(req.text), 'form': {}, 'accuracy': impact_detector.accuracy})
    except Exception as err:
        return render_template(TMP, **{'error': err, 'form': {}, 'accuracy': impact_detector.accuracy})
