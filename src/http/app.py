import importlib
from flask import Flask, render_template
from src.config import config

app = Flask(__name__)


def init():
    configure()
    installing_blueprints()


def configure():
    app.config.from_object(config)
    if config:
        app.config.from_object(config)


def installing_blueprints():
    for api in config.apps:
        a = importlib.import_module('src.http.%s' % api)
        app.register_blueprint(a.mod)
        a.ai_model.train()


@app.route('/')
def home():
    return render_template('home.html')


def run():
    init()
    app.run(debug=config.debug, host='0.0.0.0', port=int(
        config.api_port), threaded=True)
